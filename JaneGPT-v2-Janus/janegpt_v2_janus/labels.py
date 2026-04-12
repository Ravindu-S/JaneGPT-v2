"""
JaneGPT-v3 label sets (aligned to your final dataset policy)
"""

# --- Domains ---
DOMAIN_LABELS = [
    "volume",
    "brightness",
    "media",
    "apps",
    "browser",
    "productivity",
    "screen",
    "window",
    "system",
    "conversation",
]
DOMAIN_TO_ID = {d: i for i, d in enumerate(DOMAIN_LABELS)}
ID_TO_DOMAIN = {i: d for d, i in DOMAIN_TO_ID.items()}

# --- Actions ---
ACTION_LABELS = [
    # adjustments
    "up", "down", "set", "mute", "unmute",

    # media
    "play", "pause", "next", "previous",

    # apps
    "launch", "close", "switch",

    # browser/productivity/screen/conversation/system/window
    "search",
    "set_reminder",
    "screenshot",
    "read",
    "explain",
    "undo",
    "quit",
    "chat",

    # window
    "minimize", "maximize", "restore", "focus",

    # productivity
    "copy", "paste", "cut",

    # system
    "lock", "sleep",
    "wifi_on", "wifi_off",
    "bluetooth_on", "bluetooth_off",
]
ACTION_TO_ID = {a: i for i, a in enumerate(ACTION_LABELS)}
ID_TO_ACTION = {i: a for a, i in ACTION_TO_ID.items()}

# --- Allowed domain->actions (runtime enforcement + sanity checks) ---
ALLOWED_DOMAIN_ACTIONS = {
    "volume": {"up","down","set","mute","unmute"},
    "brightness": {"up","down","set"},
    "media": {"play","pause","next","previous"},
    "apps": {"launch","close","switch"},
    "browser": {"search"},
    "productivity": {"set_reminder","copy","cut","paste"},
    "screen": {"screenshot"},
    "window": {"focus","minimize","maximize","restore"},
    "system": {"undo","lock","sleep","wifi_on","wifi_off","bluetooth_on","bluetooth_off"},
    "conversation": {"chat","read","explain","quit"},
}

def is_allowed_pair(domain: str, action: str) -> bool:
    return action in ALLOWED_DOMAIN_ACTIONS.get(domain, set())

# --- Slots (BIO tagging) ---
SLOT_TYPES = [
    "VALUE",
    "APP_NAME",
    "QUERY",
    "DURATION",
    "TIME",
    "WINDOW_NAME",
    "TEXT",
]

SLOT_LABELS = ["O"]
for st in SLOT_TYPES:
    SLOT_LABELS.append(f"B-{st}")
    SLOT_LABELS.append(f"I-{st}")

SLOT_TO_ID = {s: i for i, s in enumerate(SLOT_LABELS)}
ID_TO_SLOT = {i: s for s, i in SLOT_TO_ID.items()}

# --- Required slots (for CLARIFICATION at runtime) ---
# Keep these strict only where your executor truly cannot proceed without the slot.
REQUIRED_SLOTS = {
    ("volume", "set"): ["VALUE"],
    ("brightness", "set"): ["VALUE"],
    ("apps", "launch"): ["APP_NAME"],
    ("apps", "close"): ["APP_NAME"],
    ("apps", "switch"): ["APP_NAME"],
    ("browser", "search"): ["QUERY"],
    # reminder can be TIME or DURATION: allow either
    ("productivity", "set_reminder"): ["TIME|DURATION"],
    ("window", "focus"): ["WINDOW_NAME"],
    ("conversation", "read"): ["TEXT"],
    ("conversation", "explain"): ["TEXT"],
}

# --- Legacy mapping back to v2 intent strings (optional) ---
# NOTE: v2 doesn't have copy/cut/paste/wifi/bluetooth/lock/sleep/window actions.
LEGACY_INTENT_MAP = {
    ("volume", "up"): "volume_up",
    ("volume", "down"): "volume_down",
    ("volume", "set"): "volume_set",
    ("volume", "mute"): "volume_mute",

    ("brightness", "up"): "brightness_up",
    ("brightness", "down"): "brightness_down",
    ("brightness", "set"): "brightness_set",

    ("media", "play"): "media_play",
    ("media", "pause"): "media_pause",
    ("media", "next"): "media_next",
    ("media", "previous"): "media_previous",

    ("browser", "search"): "browser_search",

    ("apps", "launch"): "app_launch",
    ("apps", "close"): "app_close",
    ("apps", "switch"): "app_switch",

    ("productivity", "set_reminder"): "set_reminder",
    ("screen", "screenshot"): "screenshot",

    # your dataset uses conversation/read/explain; map to v2 screen intents if you want that behavior
    ("conversation", "read"): "read_screen",
    ("conversation", "explain"): "explain_screen",

    ("system", "undo"): "undo",
    ("conversation", "chat"): "chat",
    ("conversation", "quit"): "quit_jane",
}

def to_legacy_intent(domain: str, action: str):
    return LEGACY_INTENT_MAP.get((domain, action))
