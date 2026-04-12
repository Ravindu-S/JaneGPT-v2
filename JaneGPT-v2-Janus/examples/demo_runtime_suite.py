import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    from runtime.jane_nlu_runtime import JaneNLURuntime

    rt = JaneNLURuntime(base_dir=str(root))

    # ─────────────────────────────────────────────
    # Test suites
    # Each entry: (description, list of turns)
    # ─────────────────────────────────────────────
    suites = {

        # ── 1. Basic single-turn commands ──────────
        "BASIC COMMANDS": [
            ("volume up",           ["increase the volume"]),
            ("volume down",         ["turn it down"]),
            ("volume set",          ["set volume to 50"]),
            ("mute",                ["mute"]),
            ("unmute",              ["unmute the audio"]),
            ("brightness up",       ["make the screen brighter"]),
            ("brightness set",      ["set brightness to 70"]),
            ("media play",          ["play music"]),
            ("media pause",         ["pause"]),
            ("media next",          ["skip this song"]),
            ("screenshot",          ["take a screenshot"]),
            ("app launch",          ["open chrome"]),
            ("app close",           ["close discord"]),
            ("app switch",          ["switch to vs code"]),
            ("browser search",      ["search for python tutorials"]),
            ("wifi on",             ["turn on wifi"]),
            ("wifi off",            ["disable wifi"]),
            ("bluetooth on",        ["enable bluetooth"]),
            ("bluetooth off",       ["turn bluetooth off"]),
            ("lock",                ["lock the pc"]),
            ("sleep",               ["put system to sleep"]),
            ("set reminder",        ["remind me in 10 minutes"]),
            ("focus window",        ["focus on task manager"]),
            ("minimize window",     ["minimize this window"]),
            ("maximize window",     ["maximize this window"]),
            ("restore window",      ["restore the window"]),
            ("copy",                ["copy the stack trace"]),
            ("paste",               ["paste here"]),
            ("cut",                 ["cut the release notes"]),
            ("read",                ["read this paragraph"]),
            ("explain",             ["explain the selected text"]),
            ("undo",                ["undo that"]),
            ("quit",                ["quit jane"]),
            ("chat",                ["hello"]),
        ],

        # ── 2. Pending slot fill (multi-turn) ──────
        "PENDING SLOT FILL": [
            ("value fill — volume set",         ["set volume", "55"]),
            ("query fill — search",             ["search", "cats on youtube"]),
            ("app name fill — launch",          ["open", "spotify"]),
            ("reminder fill — duration",        ["remind me", "30 minutes"]),
            ("reminder fill — time",            ["set a reminder", "7:30 am"]),
            ("window name fill — focus",        ["focus window", "task manager"]),
            ("text fill — read",                ["read", "the release notes"]),
        ],

        # ── 3. Follow-up / context sensitivity ─────
        "FOLLOW-UP CONTEXT": [
            ("correction after volume up",      ["increase the volume", "that is not enough", "that's too loud"]),
            ("set after vague command",         ["set volume", "55"]),
            ("hello doesn't clear state",       ["open chrome", "hello", "close chrome"]),
        ],

        # ── 4. Typo robustness ─────────────────────
        "TYPOS": [
            ("opne chrome",         ["opne chrome"]),
            ("swtich to vscode",    ["swtich to vscode"]),
            ("voluem up",           ["voluem up"]),
            ("brighness down",      ["brighness down"]),
            ("serach for cats",     ["serach for cats"]),
            ("screeenshot",         ["screeenshot"]),
        ],

        # ── 5. Real natural language (non-template) ─
        "NATURAL / OFF-SCRIPT": [
            ("informal volume",     ["yo turn that up"]),
            ("polite brightness",   ["could you please dim the lights a bit?"]),
            ("partial command",     ["volume"]),
            ("very short",          ["wifi"]),
            ("ambiguous pause",     ["stop"]),
            ("ambiguous mute",      ["kill the sound"]),
            ("mixed case",          ["OPEN CHROME"]),
            ("with punctuation",    ["open chrome!!"]),
            ("number only",         ["50"]),
            ("empty-ish",           ["   "]),
        ],

        # ── 6. Edge cases ──────────────────────────
        "EDGE CASES": [
            ("pending timeout simulation",  ["set volume"]),   # no follow-up → next turn unrelated
            ("back-to-back commands",       ["open chrome", "close chrome"]),
            ("chat mid-session",            ["open chrome", "hello there", "close chrome"]),
            ("repeat same command",         ["mute", "mute"]),
            ("contradiction",              ["turn volume up", "turn volume down"]),
        ],
    }

    print("\n" + "═" * 70)
    print("  JaneGPT-v3 Janus — Runtime Test Suite")
    print("═" * 70)

    total_turns = 0
    total_routed_local = 0
    total_routed_llama = 0
    total_clarify = 0
    failures = []

    for suite_name, cases in suites.items():
        print(f"\n{'─'*70}")
        print(f"  {suite_name}")
        print(f"{'─'*70}")

        for case_name, turns in cases:
            state = {}
            print(f"\n  ▸ {case_name}")
            for turn in turns:
                if not turn.strip():
                    turn_display = repr(turn)
                else:
                    turn_display = turn

                try:
                    out, state = rt.handle_turn(turn, state)
                    total_turns += 1

                    do = out.get("do")
                    say = out.get("say")

                    if say:
                        total_clarify += 1
                        result_str = f"[CLARIFY] {say}"
                    elif do and do.get("route_to_llama"):
                        total_routed_llama += 1
                        result_str = "[LLAMA]"
                    elif do:
                        total_routed_local += 1
                        domain = do.get("domain", "?")
                        action = do.get("action", "?")
                        slots = do.get("slots", {})
                        conf = do.get("confidence")
                        conf_str = f" ({conf:.0%})" if conf else ""
                        slot_str = ""
                        if slots:
                            slot_parts = [f"{k}={v.get('text', v)}" for k, v in slots.items()]
                            slot_str = " | " + ", ".join(slot_parts)
                        result_str = f"[LOCAL] {domain}/{action}{conf_str}{slot_str}"
                    else:
                        result_str = "[EMPTY OUTPUT]"
                        failures.append((case_name, turn, "empty output"))

                    print(f"    USER : {turn_display}")
                    print(f"    OUT  : {result_str}")

                except Exception as e:
                    failures.append((case_name, turn, str(e)))
                    print(f"    USER : {turn_display}")
                    print(f"    ERROR: {e}")

    # ── Summary ────────────────────────────────────
    print("\n" + "═" * 70)
    print("  SUMMARY")
    print("═" * 70)
    print(f"  Total turns     : {total_turns}")
    print(f"  Local commands  : {total_routed_local}")
    print(f"  Llama routes    : {total_routed_llama}")
    print(f"  Clarifications  : {total_clarify}")
    print(f"  Errors          : {len(failures)}")

    if failures:
        print("\n  FAILURES:")
        for case, turn, reason in failures:
            print(f"    [{case}] '{turn}' → {reason}")
    else:
        print("\n  ✅ No errors thrown.")

    print("\n  NOTE: Check [LLAMA] outputs — those are where the model")
    print("  deferred to Llama instead of handling locally. If any")
    print("  obvious commands ended up there, your model needs more data")
    print("  or your confidence threshold is too low.")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    main()