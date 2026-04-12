import os, re, time
from typing import Dict, Tuple, Any, Optional

from janegpt_v2_janus.inference import JaneGPTv3NLU
from janegpt_v2_janus import labels as L


# ─────────────────────────────────────────────────────────────
# Words that are NOT real window/panel names.
# If the model extracts one of these as WINDOW_NAME, discard it.
# ─────────────────────────────────────────────────────────────
_FAKE_WINDOW_NAMES = {
    "window", "this window", "the window", "a window",
    "panel", "this panel", "the panel", "a panel",
    "pane", "this pane", "the pane",
    "screen", "the screen", "this screen",
    "it", "that", "this", "here",
}


def _is_fake_window_name(name: str) -> bool:
    return name.strip().lower() in _FAKE_WINDOW_NAMES


def extract_value(text: str) -> Optional[str]:
    # Try digit first
    m = re.search(r"\b(\d{1,3})\b", text)
    if m:
        return m.group(1)
    t = text.strip()
    return t if t else None


def extract_free_text(text: str) -> Optional[str]:
    t = text.strip()
    return t if t else None


_COLLOQUIAL_TURN_UP = re.compile(r"\bturn\s+(that|it)\s+up\b", re.I)
_COLLOQUIAL_TURN_DOWN = re.compile(r"\bturn\s+(that|it)\s+down\b", re.I)


class JaneNLURuntime:
    """
    Production runtime wrapper:
    - stateful follow-ups
    - clarification -> pending slot fill
    - llama routing for chat
    """

    def __init__(
        self,
        base_dir: str,
        model_filename: str = "janegpt_v2_janus.pt",
        device=None,
    ):
        self.base_dir = base_dir
        self.weights_dir = os.path.join(base_dir, "weights")
        self.model_path = os.path.join(self.weights_dir, model_filename)
        self.tok_path = os.path.join(self.weights_dir, "tokenizer.json")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(self.model_path)
        if not os.path.exists(self.tok_path):
            raise FileNotFoundError(self.tok_path)

        self.nlu = JaneGPTv3NLU(self.model_path, self.tok_path, device=device)

    # ──────────────────────────────────────────────
    # FIX 1 helper: strip WINDOW_NAME slots that
    # contain generic non-names like "window", "pane"
    # ──────────────────────────────────────────────
    def _clean_slots(self, slots: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = {}
        for slot_type, slot_val in slots.items():
            if slot_type == "WINDOW_NAME":
                raw = slot_val.get("text", "") if isinstance(slot_val, dict) else str(slot_val)
                if _is_fake_window_name(raw):
                    continue  # drop fake window name
            cleaned[slot_type] = slot_val
        return cleaned

    def _missing_required_slot(self, domain: str, action: str, slots: Dict[str, Any]) -> Optional[str]:
        reqs = getattr(L, "REQUIRED_SLOTS", {}).get((domain, action), [])
        for req in reqs:
            if req == "TIME|DURATION":
                if ("TIME" not in slots) and ("DURATION" not in slots):
                    return req
            elif req not in slots:
                return req
        return None

    def _question_for_need(self, need: str) -> str:
        return {
            "VALUE": "What value should I set it to?",
            "APP_NAME": "Which app?",
            "QUERY": "What should I search for?",
            "WINDOW_NAME": "Which window should I focus?",
            "TEXT": "What text should I use?",
            "TIME|DURATION": "When should I set the reminder (time or duration)?",
        }.get(need, f"Please provide {need}.")

    def _colloquial_override(self, text: str, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        t = text.strip().lower()
        action = None

        if _COLLOQUIAL_TURN_UP.search(t):
            action = "up"
        elif _COLLOQUIAL_TURN_DOWN.search(t):
            action = "down"

        if action is None:
            return None

        # Resolve target control domain with strong lexical hints first.
        if any(k in t for k in ("brightness", "screen", "light", "lights")):
            domain = "brightness"
        elif any(k in t for k in ("volume", "sound", "audio", "loud", "quiet")):
            domain = "volume"
        else:
            last_domain = state.get("last_domain")
            domain = last_domain if last_domain in ("volume", "brightness") else "volume"

        legacy = L.to_legacy_intent(domain, action) if hasattr(L, "to_legacy_intent") else None
        return {
            "type": "command",
            "domain": domain,
            "action": action,
            "slots": {},
            "confidence": 1.0,
            "legacy_intent": legacy,
            "route": "local",
            "notes": {"colloquial_override": True},
        }

    def _resolve_pending(
        self,
        text: str,
        state: Dict[str, Any],
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:

        pending = state.get("pending")
        if not pending:
            return None, state

        # optional timeout support
        ttl = pending.get("ttl_sec")
        ts = pending.get("ts")
        if ttl is not None and ts is not None:
            if time.time() - ts > ttl:
                state = dict(state)
                state.pop("pending", None)
                return None, state

        domain = pending["domain"]
        action = pending["action"]
        need   = pending["need"]
        if isinstance(need, str) and need.lower() == "time_or_duration":
            need = "TIME|DURATION"

        slots = {}
        t = text.strip()

        if need == "VALUE":
            v = extract_value(t)
            if v is None:
                return None, state
            slots["VALUE"] = {"text": v, "confidence": 1.0, "start": None, "end": None}

        elif need in ("QUERY", "APP_NAME", "WINDOW_NAME", "TEXT"):
            v = extract_free_text(t)
            if v is None:
                return None, state
            slots[need] = {"text": v, "confidence": 1.0, "start": None, "end": None}

        elif need == "TIME|DURATION":
            v = extract_free_text(t)
            if v is None:
                return None, state

            # FIX 2: Improved TIME detection.
            # Previously "7:30 am" could fall through because the
            # pending state was not being stored when the clarify
            # response was generated. Now we also widen the TIME
            # pattern to catch more formats (e.g. "7:30 am",
            # "noon", "midnight", "tomorrow at 9 am", "friday 5 pm").
            time_pattern = re.compile(
                r"""
                \b(
                    \d{1,2}:\d{2}           # 7:30 or 14:20
                    |\d{1,2}\s*(am|pm)      # 7am / 7 am / 11pm
                    |noon|midnight          # keywords
                    |tomorrow               # relative
                    |(next\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)
                )\b
                """,
                re.VERBOSE | re.IGNORECASE,
            )
            if time_pattern.search(t):
                slots["TIME"] = {"text": v, "confidence": 1.0, "start": None, "end": None}
            else:
                slots["DURATION"] = {"text": v, "confidence": 1.0, "start": None, "end": None}
        else:
            return None, state

        # clear pending
        new_state = dict(state)
        new_state.pop("pending", None)

        legacy = L.to_legacy_intent(domain, action) if hasattr(L, "to_legacy_intent") else None

        resolved = {
            "type": "command",
            "domain": domain,
            "action": action,
            "slots": slots,
            "confidence": 1.0,
            "legacy_intent": legacy,
            "route": "local",
            "notes": {"resolved_from_pending": True},
        }
        return resolved, new_state

    def handle_turn(
        self,
        user_text: str,
        state: Optional[Dict[str, Any]] = None,
        pending_ttl_sec: int = 30,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Returns: (output, new_state)

        output:
          - {"say": "...", "do": None}                              clarification
          - {"say": None, "do": {"route_to_llama": True, ...}}     chat / unknown
          - {"say": None, "do": {"domain": ..., "action": ...}}    local command
        """
        state = dict(state or {})

        # ── FIX 3: Guard against empty / whitespace-only input ──
        if not user_text or not user_text.strip():
            return {
                "say": "I didn't catch that. Could you repeat?",
                "do": None,
            }, state

        # ── 1) Resolve pending slot fill ────────────────────────
        resolved, state = self._resolve_pending(user_text, state)
        if resolved is not None:
            slots = self._clean_slots(resolved.get("slots", {}))
            missing = self._missing_required_slot(resolved["domain"], resolved["action"], slots)
            if missing:
                state["pending"] = {
                    "domain": resolved["domain"],
                    "action": resolved["action"],
                    "need": missing,
                    "ts": time.time(),
                    "ttl_sec": pending_ttl_sec,
                }
                return {"say": self._question_for_need(missing), "do": None}, state

            resolved_clean = dict(resolved)
            resolved_clean["slots"] = slots
            state = self.nlu.update_state(resolved_clean, state)
            return {
                "say": None,
                "do": {
                    "route": "local",
                    "legacy_intent": resolved.get("legacy_intent"),
                    "domain": resolved["domain"],
                    "action": resolved["action"],
                    "slots": slots,
                    "confidence": resolved.get("confidence", 1.0),
                },
            }, state

        # ── 1b) Colloquial controls (safe narrow override) ─────
        ov = self._colloquial_override(user_text, state)
        if ov is not None:
            state = self.nlu.update_state(ov, state)
            return {
                "say": None,
                "do": {
                    "route": "local",
                    "legacy_intent": ov.get("legacy_intent"),
                    "domain": ov["domain"],
                    "action": ov["action"],
                    "slots": ov.get("slots", {}),
                    "confidence": ov.get("confidence", 1.0),
                },
            }, state

        # ── 2) Normal NLU ────────────────────────────────────────
        r = self.nlu.predict(user_text, state=state)

        # ── 2a) Clarification → store pending ───────────────────
        if r["type"] == "clarify":
            dbg    = r.get("debug", {})
            dom    = dbg.get("domain")
            act    = dbg.get("action")
            reason = dbg.get("reason", "")

            need = None
            if reason == "missing_time_or_duration":
                need = "TIME|DURATION"
            elif reason.startswith("missing_"):
                need = reason.replace("missing_", "")
                if need == "time_or_duration":
                    need = "TIME|DURATION"

            # FIX 2 cont: Only store pending when we have all
            # three pieces. Previously a missing dom/act meant
            # the pending was silently dropped, causing the next
            # turn (e.g. "7:30 am") to skip _resolve_pending
            # entirely and go straight to the NLU which then
            # routed it to Llama.
            if dom and act and need:
                state["pending"] = {
                    "domain":  dom,
                    "action":  act,
                    "need":    need,
                    "ts":      time.time(),
                    "ttl_sec": pending_ttl_sec,
                }

            return {"say": r["question"], "do": None}, state

        # ── 2b) Llama route (preserve last control state) ───────
        if r.get("route") == "llama":
            return {"say": None, "do": {"route_to_llama": True, "text": user_text}}, state

        # ── 2c) Local command → clean slots → update state ──────
        r_slots = self._clean_slots(r.get("slots", {}))
        missing = self._missing_required_slot(r["domain"], r["action"], r_slots)
        if missing:
            state["pending"] = {
                "domain": r["domain"],
                "action": r["action"],
                "need": missing,
                "ts": time.time(),
                "ttl_sec": pending_ttl_sec,
            }
            return {"say": self._question_for_need(missing), "do": None}, state

        r_for_state = dict(r)
        r_for_state["slots"] = r_slots
        state = self.nlu.update_state(r_for_state, state)

        return {
            "say": None,
            "do": {
                "route": "local",
                "legacy_intent": r.get("legacy_intent"),
                "domain": r["domain"],
                "action": r["action"],
                "slots": r_slots,
                "confidence": r.get("confidence"),
            },
        }, state


# ── Standalone entry point ───────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    from jane_nlu_runtime import JaneNLURuntime  # relative import for direct run

    rt = JaneNLURuntime(base_dir=str(root))
    state = {}

    dialog = [
        "increase the volume",
        "that is not enough",
        "that's too loud",
        "set volume",
        "55",
        "search",
        "cats",
        "set a reminder",
        "7:30 am",
        "hello",
        "   ",
    ]

    print("== JaneGPT-v2-Janus runtime demo ==")
    for u in dialog:
        out, state = rt.handle_turn(u, state)
        print(f"\nUSER : {u!r}")
        print(f"OUT  : {out}")
        print(f"STATE: {state}")