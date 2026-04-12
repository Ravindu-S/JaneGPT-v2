import sys
from pathlib import Path

def main():
    # repo root = one level above /examples
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))  # so `import janegpt_v2_janus` works

    model_path = root / "weights" / "janegpt_v2_janus.pt"
    tok_path   = root / "weights" / "tokenizer.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not tok_path.exists():
        raise FileNotFoundError(f"Missing tokenizer file: {tok_path}")

    # If you used Option A alias in janegpt_v2_janus/__init__.py:
    from janegpt_v2_janus import JaneGPTJanusNLU

    nlu = JaneGPTJanusNLU(
        model_path=str(model_path),
        tokenizer_path=str(tok_path),
    )

    state = {}
    tests = [
        "increase the volume",
        "that is not enough",
        "that's too loud",
        "set volume",
        "55",
        "open chrome",
        "search",
        "cats",
        "hello",
        "thanks",
    ]

    print("== JaneGPT-v2-Janus demo ==")
    for t in tests:
        r = nlu.predict(t, state=state)
        print("\nUSER:", t)
        print("OUT :", r)
        if r.get("type") == "command":
            state = nlu.update_state(r, state)

if __name__ == "__main__":
    main()