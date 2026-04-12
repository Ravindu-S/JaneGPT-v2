import sys
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    # runtime wrapper (pending slot fill + follow-ups + llama routing)
    from runtime.jane_nlu_runtime import JaneNLURuntime

    rt = JaneNLURuntime(base_dir=str(root))
    state = {}

    dialog = [
        "increase the volume",
        "that is not enough",
        "that's too loud",
        "set volume",
        "55",          # <- will now resolve pending VALUE
        "search",
        "cats",        # <- will now resolve pending QUERY
        "hello",       # <- routes to llama without destroying control state
    ]

    print("== JaneGPT-v2-Janus runtime demo ==")
    for u in dialog:
        out, state = rt.handle_turn(u, state)
        print("\nUSER:", u)
        print("OUT :", out)
        print("STATE:", state)

if __name__ == "__main__":
    main()