# tools/generate_janus_report.py
import argparse, json, os, time
from pathlib import Path

import torch
import numpy as np

# Optional (only for tokenizer + dataset eval)
from tokenizers import Tokenizer

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

@torch.no_grad()
def eval_val(model, val_loader, device, slot_o_id=0):
    model.eval()
    total = 0
    cd = ca = cp = 0
    loss_sum = 0.0
    steps = 0

    ignore = -100
    tp = fp = fn = 0

    for b in val_loader:
        x = b["input_ids"].to(device, non_blocking=True)
        m = b["attention_mask"].to(device, non_blocking=True)
        y_dom = b["labels_domain"].to(device, non_blocking=True)
        y_act = b["labels_action"].to(device, non_blocking=True)
        y_slots = b["labels_slots"].to(device, non_blocking=True)

        out = model(x, attention_mask=m, labels_domain=y_dom, labels_action=y_act, labels_slots=y_slots, causal=False)
        loss_sum += float(out["loss"].item())
        steps += 1

        dp = out["logits_domain"].argmax(-1)
        ap = out["logits_action"].argmax(-1)

        total += y_dom.size(0)
        cd += (dp == y_dom).sum().item()
        ca += (ap == y_act).sum().item()
        cp += ((dp == y_dom) & (ap == y_act)).sum().item()

        sp = out["logits_slots"].argmax(-1)
        mask = y_slots.ne(ignore)
        gold = y_slots[mask]
        pred = sp[mask]

        pred_pos = pred.ne(slot_o_id)
        gold_pos = gold.ne(slot_o_id)

        tp += ((pred == gold) & gold_pos).sum().item()
        fp += (pred_pos & ~gold_pos).sum().item() + (pred_pos & gold_pos & (pred != gold)).sum().item()
        fn += (~pred_pos & gold_pos).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "val_loss": loss_sum / max(steps, 1),
        "domain_acc": cd / max(total, 1),
        "action_acc": ca / max(total, 1),
        "pair_acc": cp / max(total, 1),
        "slot_precision": precision,
        "slot_recall": recall,
        "slot_f1": f1,
        "val_examples": total,
    }

def benchmark_forward(model, tok, device, max_len, texts, iters=200, warmup=50):
    # forward-only benchmark (no decoding)
    model.eval()

    def encode(text):
        enc = tok.encode(text)
        ids = enc.ids[:max_len]
        attn = [1] * len(ids)
        pad = max_len - len(ids)
        if pad > 0:
            ids += [0] * pad
            attn += [0] * pad
        x = torch.tensor([ids], dtype=torch.long, device=device)
        m = torch.tensor([attn], dtype=torch.long, device=device)
        return x, m

    # warmup
    for i in range(warmup):
        x, m = encode(texts[i % len(texts)])
        _ = model(x, attention_mask=m, causal=False)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    times = []
    for i in range(iters):
        x, m = encode(texts[i % len(texts)])
        t0 = time.perf_counter()
        _ = model(x, attention_mask=m, causal=False)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    arr = np.array(times, dtype=np.float32)
    return {
        "forward_ms_mean": float(arr.mean()),
        "forward_ms_p50": float(np.percentile(arr, 50)),
        "forward_ms_p95": float(np.percentile(arr, 95)),
        "iters": iters,
        "warmup": warmup,
        "max_len": max_len,
        "batch_size": 1,
    }

def benchmark_predict(nlu, device, texts, iters=200, warmup=50):
    # end-to-end predict benchmark (includes domain/action/slot decode and rules)
    # use empty state for fairness
    for i in range(warmup):
        _ = nlu.predict(texts[i % len(texts)], state={})
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    times = []
    for i in range(iters):
        t = texts[i % len(texts)]
        t0 = time.perf_counter()
        _ = nlu.predict(t, state={})
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    arr = np.array(times, dtype=np.float32)
    return {
        "predict_ms_mean": float(arr.mean()),
        "predict_ms_p50": float(np.percentile(arr, 50)),
        "predict_ms_p95": float(np.percentile(arr, 95)),
        "iters": iters,
        "warmup": warmup,
        "batch_size": 1,
    }

def to_markdown(report: dict) -> str:
    m = []
    m.append("## Model Details (auto-generated)\n")
    m.append(f"- **Checkpoint**: `{report['checkpoint_path']}`")
    m.append(f"- **Checkpoint size**: `{report['checkpoint_size_mb']:.2f} MB`")
    m.append(f"- **Device used for benchmark**: `{report['device']}`")
    m.append("")
    m.append("### Architecture\n")
    cfg = report.get("config", {})
    m.append("| Property | Value |")
    m.append("|---|---:|")
    for k in ["vocab_size","embed_dim","num_layers","num_heads","num_kv_heads","ff_hidden","max_len","causal"]:
        if k in cfg:
            m.append(f"| {k} | {cfg[k]} |")
    m.append("")
    m.append("### Parameters\n")
    m.append("| Component | Params |")
    m.append("|---|---:|")
    m.append(f"| Total | {report['params_total']:,} |")
    m.append(f"| Trainable | {report['params_trainable']:,} |")
    if "params_backbone" in report:
        m.append(f"| Backbone | {report['params_backbone']:,} |")
    if "params_heads" in report:
        m.append(f"| Heads (domain+action+slots) | {report['params_heads']:,} |")
    m.append("")
    m.append("### Label Set Sizes\n")
    m.append("| Label set | Count |")
    m.append("|---|---:|")
    m.append(f"| Domains | {report['num_domains']} |")
    m.append(f"| Actions | {report['num_actions']} |")
    m.append(f"| Slot labels (BIO) | {report['num_slot_labels']} |")
    m.append("")
    if "val_metrics" in report:
        v = report["val_metrics"]
        m.append("### Validation Metrics\n")
        m.append("| Metric | Value |")
        m.append("|---|---:|")
        for k in ["val_loss","domain_acc","action_acc","pair_acc","slot_precision","slot_recall","slot_f1","val_examples"]:
            if k in v:
                m.append(f"| {k} | {v[k]:.6f} |" if isinstance(v[k], float) else f"| {k} | {v[k]} |")
        m.append("")
    if "benchmark_forward" in report:
        b = report["benchmark_forward"]
        m.append("### Inference Benchmark (forward-only)\n")
        m.append("| Stat | ms |")
        m.append("|---|---:|")
        m.append(f"| mean | {b['forward_ms_mean']:.3f} |")
        m.append(f"| p50 | {b['forward_ms_p50']:.3f} |")
        m.append(f"| p95 | {b['forward_ms_p95']:.3f} |")
        m.append("")
    if "benchmark_predict" in report:
        b = report["benchmark_predict"]
        m.append("### Inference Benchmark (end-to-end predict)\n")
        m.append("| Stat | ms |")
        m.append("|---|---:|")
        m.append(f"| mean | {b['predict_ms_mean']:.3f} |")
        m.append(f"| p50 | {b['predict_ms_p50']:.3f} |")
        m.append(f"| p95 | {b['predict_ms_p95']:.3f} |")
        m.append("")
    return "\n".join(m)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", type=str, default=".", help="Path to repo root")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to .pt checkpoint")
    ap.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer.json")
    ap.add_argument("--val", type=str, default="", help="Optional: val.jsonl for metrics")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    sys_path_added = str(repo_root)
    import sys
    if sys_path_added not in sys.path:
        sys.path.insert(0, sys_path_added)

    # import your package (adjust if your package folder is different)
    from janegpt_v2_janus.multitask import JaneGPTv3MultiTask
    from janegpt_v2_janus import labels as L
    from janegpt_v2_janus.inference import JaneGPTv3NLU  # (or alias in __init__)

    device = args.device
    ckpt_path = Path(args.ckpt).resolve()
    tok_path = Path(args.tokenizer).resolve()

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    cfg = ckpt.get("config", {})
    max_len = int(cfg.get("max_len", 96))

    model = JaneGPTv3MultiTask(
        num_domains=len(L.DOMAIN_LABELS),
        num_actions=len(L.ACTION_LABELS),
        num_slot_labels=len(L.SLOT_LABELS),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    # param counts
    params_total, params_trainable = count_params(model)
    params_backbone = sum(p.numel() for n, p in model.named_parameters() if n.startswith("backbone."))
    params_heads = params_total - params_backbone

    # checkpoint size
    size_mb = ckpt_path.stat().st_size / (1024 * 1024)

    tok = Tokenizer.from_file(str(tok_path))

    report = {
        "model_name": "JaneGPT-v2-Janus",
        "checkpoint_path": str(ckpt_path),
        "checkpoint_size_mb": size_mb,
        "device": device,
        "config": {
            "vocab_size": 8192,
            "embed_dim": 256,
            "num_layers": 8,
            "num_heads": 8,
            "num_kv_heads": 4,
            "ff_hidden": 672,
            "max_len": max_len,
            "causal": False,
        },
        "params_total": int(params_total),
        "params_trainable": int(params_trainable),
        "params_backbone": int(params_backbone),
        "params_heads": int(params_heads),
        "num_domains": len(L.DOMAIN_LABELS),
        "num_actions": len(L.ACTION_LABELS),
        "num_slot_labels": len(L.SLOT_LABELS),
    }

    # Optional validation metrics if val file exists
    if args.val and Path(args.val).exists():
        from janegpt_v2_janus.dataset import JaneGPTv3Dataset
        from torch.utils.data import DataLoader

        val_ds = JaneGPTv3Dataset(args.val, tok, max_len=max_len)
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=device.startswith("cuda"))
        report["val_metrics"] = eval_val(model, val_loader, device, slot_o_id=L.SLOT_TO_ID["O"])

    # Benchmarks
    texts = [
        "increase the volume",
        "set volume to 55%",
        "open chrome",
        "search for cats",
        "set a reminder for 5 minutes",
        "focus vscode",
        "undo",
        "hello",
        "thanks",
    ]
    report["benchmark_forward"] = benchmark_forward(model, tok, device, max_len, texts, iters=args.iters, warmup=max(20, args.iters//10))

    # end-to-end predict benchmark
    nlu = JaneGPTv3NLU(model_path=str(ckpt_path), tokenizer_path=str(tok_path), device=device)
    report["benchmark_predict"] = benchmark_predict(nlu, device, texts, iters=args.iters, warmup=max(20, args.iters//10))

    # write outputs
    out_dir = repo_root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "janus_model_report.json"
    md_path = out_dir / "janus_model_report.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(to_markdown(report))

    print("Wrote:", json_path)
    print("Wrote:", md_path)
    print("\n--- Markdown snippet preview ---\n")
    print(to_markdown(report)[:1500])

if __name__ == "__main__":
    main()