import asyncio
import io
import json
import re
import time
import logging
import os

try:
    import winsound  # Windows beep (not on Linux/cloud)
except ImportError:
    winsound = None

import pytesseract
from PIL import Image, ImageOps
from telethon import TelegramClient, events, errors, utils
from telethon.tl.functions.messages import GetDiscussionMessageRequest
from telethon.tl.functions.channels import GetFullChannelRequest
from telethon.sessions import StringSession

logging.basicConfig(level=logging.WARNING)   # shows Telethon warnings (server errors, flood waits) in the log

# ---------- SETTINGS ----------
API_ID = int(os.environ.get("API_ID", 12345))   # from my.telegram.org
API_HASH = os.environ.get("API_HASH", "your_api_hash")
CHANNEL = os.environ.get("CHANNEL", "channel_username")   # channel to watch (without @)

DATA_FILE = "data.json"            # symbols with high/low
ALIASES_FILE = "aliases.json"      # {"gold": "XAUUSD", "us tech 100": "NAS100", ...}

# Crop box of the question text as FRACTIONS of the image: (left, top, right, bottom)
# Measured on the 720x1280 question image: text is x 35-678, y 324-401 -> box (20, 315, 700, 410).
# Fractions make it work for any image size of the same layout.
CROP_FRAC = (20 / 720, 315 / 1280, 700 / 720, 410 / 1280)
LOOKUP_TIMEOUT = 0.5               # seconds before a slow discussion lookup is retried (Telethon itself waits 2s)
UPSCALE = 1                        # 1 = fastest; 2 gave no better result on this layout

TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.name == "nt":  # on Linux (cloud) tesseract is already on PATH
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
# --psm 6 = one block of text. No character whitelist: it deleted the spaces between words.
OCR_CONFIG = "--oem 1 --psm 6"
# ------------------------------


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f, parse_float=str)   # keep numbers exactly as written: 1.47850 stays "1.47850"


# Load everything ONCE at startup (in memory = fast)
raw = load_json(DATA_FILE)
if isinstance(raw, list):
    DATA = {str(item["symbol"]).upper(): item for item in raw}
else:
    DATA = {str(k).upper(): v for k, v in raw.items()}

ALIASES = {k.upper(): v.upper() for k, v in load_json(ALIASES_FILE).items()}

# Every name we can recognise -> real symbol. Longest first so
# "US TECH 100" is tried before shorter names.
NAMES = {s: s for s in DATA}
NAMES.update(ALIASES)
PATTERNS = [
    (re.compile(r"(?<![A-Z0-9])" + re.escape(name) + r"(?![A-Z0-9])"), sym)
    for name, sym in sorted(NAMES.items(), key=lambda x: -len(x[0]))
]

answered = set()  # post ids already answered (comment only once)


def alert(text):
    print(f"[!] {text}")
    if winsound:
        winsound.Beep(1200, 400)
        winsound.Beep(900, 400)


def ocr_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    l, t, r, b = CROP_FRAC
    img = img.crop((int(l * w), int(t * h), int(r * w), int(b * h)))
    img = ImageOps.invert(ImageOps.grayscale(img))   # dark text on white reads best
    if UPSCALE != 1:
        img = img.resize((img.width * UPSCALE, img.height * UPSCALE))
    return pytesseract.image_to_string(img, config=OCR_CONFIG)


def find_symbol(text):
    text = re.sub(r"\s+", " ", text.upper())
    for pattern, symbol in PATTERNS:
        if pattern.search(text) and symbol in DATA:
            return symbol
    # fallback: OCR glued words together ("EXNESSXAUUSDCHART") -> search without spaces
    flat = re.sub(r"[^A-Z0-9]", "", text)
    for name, symbol in sorted(NAMES.items(), key=lambda x: -len(x[0])):
        flat_name = re.sub(r"[^A-Z0-9]", "", name)
        if len(flat_name) >= 5 and flat_name in flat and symbol in DATA:
            return symbol
    return None


def build_answer(symbol, text):
    # only the number(s): no symbol name, no "High:" / "Low:" words
    item = DATA[symbol]
    t = text.upper()
    wants_high = "HIGH" in t
    wants_low = "LOW" in t
    if wants_high and not wants_low:
        return f"{item['high']}"
    if wants_low and not wants_high:
        return f"{item['low']}"
    return f"{item['high']}\n{item['low']}"   # both asked/unclear: high on line 1, low on line 2


SESSION = os.environ.get("SESSION")  # string session (cloud); empty = local .session file
client = TelegramClient(StringSession(SESSION) if SESSION else "auto_answer_session", API_ID, API_HASH)


CHANNEL_INPUT = None   # resolved once at startup
CHANNEL_ID = None      # numeric id of the channel (to recognise its copies in the group)
GROUP_INPUT = None     # linked discussion group (input peer), resolved once at startup

# post id in channel -> Future that receives (group_peer, message_id_in_group)
thread_futs = {}


def thread_future(post_id):
    fut = thread_futs.get(post_id)
    if fut is None:
        fut = asyncio.get_running_loop().create_future()
        thread_futs[post_id] = fut
    return fut


def prune_futures():
    # keep memory small: forget the oldest entries if too many pile up
    if len(thread_futs) > 300:
        for k in sorted(thread_futs)[:150]:
            thread_futs.pop(k, None)


async def on_group_copy(event):
    """The channel post appeared in the discussion group -> that message is the comment target."""
    m = event.message
    fwd = m.fwd_from
    if not fwd or not fwd.channel_post:
        return
    src = getattr(fwd.from_id, "channel_id", None)
    if CHANNEL_ID is not None and src is not None and src != CHANNEL_ID:
        return
    fut = thread_future(fwd.channel_post)
    if not fut.done():
        fut.set_result((GROUP_INPUT, m.id))
    prune_futures()


async def get_discussion_target(post_id, give_up=10.0):
    """Find the comment thread of a post. Telethon sleeps 2s after every Telegram server error,
    which is what made 'send' take 2.4s / 4.5s. Here a slow try is cancelled and repeated at once."""
    start = time.perf_counter()
    tries = 0
    last_err = None
    while time.perf_counter() - start < give_up:
        tries += 1
        try:
            r = await asyncio.wait_for(
                client(GetDiscussionMessageRequest(peer=CHANNEL_INPUT, msg_id=post_id)),
                timeout=LOOKUP_TIMEOUT,
            )
            m = r.messages[0]
            chat = next(c for c in r.chats if c.id == m.peer_id.channel_id)
            if tries > 1:
                print(f"[debug] lookup found thread after {tries} tries, last error: {last_err!r}")
            return utils.get_input_peer(chat), m.id
        except asyncio.TimeoutError:
            last_err = "timeout"         # Telethon was sleeping on a server error -> retry now
        except errors.RPCError as e:
            last_err = e
            await asyncio.sleep(0.05)    # post's comment thread not ready yet
    return None


async def resolve_target(post_id):
    """Race: group listener vs. direct lookup. Whoever gets the thread first wins."""
    fut = thread_future(post_id)
    if fut.done():
        print("[thread] ready before needed (group event)")
        thread_futs.pop(post_id, None)
        return fut.result()
    poll = asyncio.create_task(get_discussion_target(post_id))
    try:
        done, _ = await asyncio.wait({fut, poll}, return_when=asyncio.FIRST_COMPLETED)
        if fut in done:
            print("[thread] won by group event")
            return fut.result()
        res = poll.result()
        if res:
            print("[thread] won by lookup")
            return res
        # lookup gave up: wait a little longer for the group event
        try:
            return await asyncio.wait_for(asyncio.shield(fut), 5)
        except asyncio.TimeoutError:
            return None
    finally:
        poll.cancel()
        thread_futs.pop(post_id, None)


@client.on(events.NewMessage(chats=CHANNEL))
async def handler(event):
    msg = event.message
    if not msg.photo:
        if msg.text and not msg.document:
            print("txt ignored")
        return
    if msg.id in answered:
        return
    answered.add(msg.id)  # mark first so we never comment twice

    t0 = time.perf_counter()
    # start looking for the comment thread NOW, while we download + OCR the image
    target_task = asyncio.create_task(resolve_target(msg.id))

    image_bytes = await msg.download_media(bytes)   # in memory, no disk
    t1 = time.perf_counter()

    text = await asyncio.to_thread(ocr_image, image_bytes)
    t2 = time.perf_counter()

    symbol = find_symbol(text)
    if not symbol:
        target_task.cancel()
        alert(f"Could not answer post {msg.id}. OCR read: {text.strip()!r}")
        return

    answer = build_answer(symbol, text)
    target = await target_task
    t3 = time.perf_counter()
    if target:
        peer, reply_id = target
        await client.send_message(peer, answer, reply_to=reply_id)
    else:   # fallback: the slow built-in way
        await client.send_message(CHANNEL, answer, comment_to=msg)
    t4 = time.perf_counter()

    print(f"{symbol} | download {t1-t0:.2f}s | OCR {t2-t1:.2f}s | wait-thread {t3-t2:.2f}s | send {t4-t3:.2f}s | total {t4-t0:.2f}s")


async def main():
    await client.start()
    global CHANNEL_INPUT, CHANNEL_ID, GROUP_INPUT

    me = await client.get_me()
    me_name = f"@{me.username}" if me.username else (me.first_name or str(me.id))
    print(f"Account: {me_name}")

    CHANNEL_INPUT = await client.get_input_entity(CHANNEL)   # resolve once, not per post
    entity = await client.get_entity(CHANNEL_INPUT)
    CHANNEL_ID = entity.id
    name = getattr(entity, "title", None) or CHANNEL
    username = f" (@{entity.username})" if getattr(entity, "username", None) else ""
    print(f"Watching channel: {name}{username} ... (Ctrl+C to stop)")

    try:
        full = await client(GetFullChannelRequest(CHANNEL_INPUT))
        linked_id = full.full_chat.linked_chat_id
        if linked_id:
            group = await client.get_entity(linked_id)
            GROUP_INPUT = utils.get_input_peer(group)
            g_user = f" (@{group.username})" if getattr(group, "username", None) else ""
            print(f"Comments go to group: {getattr(group, 'title', linked_id)}{g_user}")
            # listen to the group: its copy of each channel post is the fastest way to know the target
            client.add_event_handler(on_group_copy, events.NewMessage(chats=group))
            print("Listening to the group for instant comment targets")
        else:
            print("[!] This channel has no linked discussion group, so comments will fail")
    except Exception as e:
        print(f"[!] Could not check the linked group: {e}")

    await client.run_until_disconnected()


if __name__ == "__main__":
    client.loop.run_until_complete(main())
