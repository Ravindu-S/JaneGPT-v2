"""
Display detailed information about JaneGPT v2 model.

Shows architecture, parameters, training info, and size comparisons.
"""

import os
import torch
from model.architecture import JaneGPTv2Classifier, INTENT_LABELS


def main():
    # Load checkpoint
    checkpoint_path = "weights/janegpt_v2_classifier.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get('config', {})

    # Create model
    model = JaneGPTv2Classifier(
        vocab_size=config.get('vocab_size', 8192),
        embed_dim=config.get('embed_dim', 256),
        num_heads=config.get('num_heads', 8),
        num_kv_heads=config.get('num_kv_heads', 4),
        num_layers=config.get('num_layers', 8),
        ff_hidden=config.get('ff_hidden', 672),
        max_seq_len=config.get('max_seq_len', 256),
        dropout=config.get('dropout', 0.1),
        rope_theta=config.get('rope_theta', 10000.0),
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffers = sum(b.numel() for b in model.buffers())

    print("=" * 60)
    print("  JANEGPT v2 - MODEL INFORMATION")
    print("=" * 60)

    # Architecture
    print("\n  ARCHITECTURE")
    print(f"    Type:               Decoder-only Transformer (Classifier)")
    print(f"    Vocab Size:         {config.get('vocab_size', 8192):,}")
    print(f"    Embedding Dim:      {config.get('embed_dim', 256)}")
    print(f"    Attention Heads:    {config.get('num_heads', 8)}")
    print(f"    KV Heads (GQA):     {config.get('num_kv_heads', 4)}")
    print(f"    Head Dim:           {config.get('embed_dim', 256) // config.get('num_heads', 8)}")
    print(f"    Layers:             {config.get('num_layers', 8)}")
    print(f"    FF Hidden:          {config.get('ff_hidden', 672)}")
    print(f"    Max Seq Length:     {config.get('max_seq_len', 256)}")
    print(f"    Dropout:            {config.get('dropout', 0.1)}")
    print(f"    RoPE Theta:         {config.get('rope_theta', 10000.0)}")

    # Features
    print("\n  FEATURES")
    print(f"    Position Encoding:  RoPE (Rotary Position Embedding)")
    print(f"    Normalization:      RMSNorm")
    print(f"    Attention:          Grouped Query Attention (GQA)")
    print(f"    Feed-Forward:       SwiGLU")
    print(f"    Classifier Head:    Linear -> GELU -> Dropout -> Linear")
    print(f"    Output Classes:     {len(INTENT_LABELS)}")

    # Parameters
    print("\n  PARAMETERS")
    print(f"    Total Parameters:       {total_params:>12,}")
    print(f"    Trainable Parameters:   {trainable_params:>12,}")
    print(f"    Non-trainable Buffers:  {buffers:>12,}")
    print(f"    Model Size (float32):   {total_params * 4 / 1024 / 1024:.2f} MB")
    print(f"    Model Size (float16):   {total_params * 2 / 1024 / 1024:.2f} MB")

    # Breakdown
    print("\n  PARAMETER BREAKDOWN")
    print(f"    {'Component':<35} {'Params':>12} {'%':>8}")
    print(f"    {'-' * 55}")

    emb_params = sum(p.numel() for p in model.token_embedding.parameters())
    print(f"    {'Token Embedding':<35} {emb_params:>12,} {emb_params/total_params*100:>7.1f}%")

    all_layers_params = sum(p.numel() for p in model.layers.parameters())
    print(f"    {'Transformer Layers (total)':<35} {all_layers_params:>12,} {all_layers_params/total_params*100:>7.1f}%")

    # Single layer breakdown
    layer0_params = sum(p.numel() for p in model.layers[0].parameters())
    attn_params = sum(p.numel() for p in model.layers[0].attn.parameters()) - sum(
        b.numel() for b in model.layers[0].attn.buffers()
    )
    ff_params = sum(p.numel() for p in model.layers[0].ff.parameters())
    norm_params = model.layers[0].norm1.weight.numel() + model.layers[0].norm2.weight.numel()

    print(f"      {'  Per layer (x8):':<33} {layer0_params:>12,}")
    print(f"      {'    Attention (Q/K/V/Out)':<33} {attn_params:>12,}")
    print(f"      {'    Feed-Forward (SwiGLU)':<33} {ff_params:>12,}")
    print(f"      {'    Norms (RMSNorm x2)':<33} {norm_params:>12,}")

    final_norm_params = model.norm.weight.numel()
    print(f"    {'Final RMSNorm':<35} {final_norm_params:>12,} {final_norm_params/total_params*100:>7.1f}%")

    head_params = sum(p.numel() for p in model.intent_head.parameters())
    print(f"    {'Classification Head':<35} {head_params:>12,} {head_params/total_params*100:>7.1f}%")
    print(f"      {'  Linear(256, 256) + bias':<33} {256 * 256 + 256:>12,}")
    print(f"      {'  Linear(256, 22) + bias':<33} {256 * 22 + 22:>12,}")

    # Training
    print("\n  TRAINING")
    print(f"    Best Val Accuracy:  {checkpoint.get('val_acc', 0):.2f}%")
    print(f"    Best Val Loss:      {checkpoint.get('val_loss', 0):.4f}")
    print(f"    Best Epoch:         {checkpoint.get('epoch', 'N/A')}")

    # Intent classes
    print(f"\n  INTENT CLASSES ({len(INTENT_LABELS)})")
    for i, label in enumerate(INTENT_LABELS):
        print(f"    {i:>2}: {label}")

    # File info
    print(f"\n  FILES")
    if os.path.exists(checkpoint_path):
        model_size = os.path.getsize(checkpoint_path)
        print(f"    Checkpoint:   {model_size / 1024 / 1024:.2f} MB")

    tokenizer_path = "weights/tokenizer.json"
    if os.path.exists(tokenizer_path):
        tok_size = os.path.getsize(tokenizer_path)
        print(f"    Tokenizer:    {tok_size / 1024:.1f} KB")

    # Size comparison
    print(f"\n  SIZE COMPARISON")
    print(f"    {'Model':<30} {'Parameters':>15} {'Size':>10}")
    print(f"    {'-' * 55}")
    print(f"    {'JaneGPT v2 (this model)':<30} {total_params:>12,}   {total_params * 4 / 1024 / 1024:>5.1f} MB")
    print(f"    {'DistilBERT':<30} {'66,000,000':>15} {'260.0 MB':>10}")
    print(f"    {'BERT Base':<30} {'110,000,000':>15} {'440.0 MB':>10}")
    print(f"    {'GPT-2 Small':<30} {'124,000,000':>15} {'500.0 MB':>10}")
    print(f"    {'Llama 3 8B':<30} {'8,000,000,000':>15} {'  16.0 GB':>10}")
    print(f"    {'GPT-4':<30} {'~1,800,000,000,000':>15} {'~  3.6 TB':>10}")

    print(f"\n  Created by: Ravindu Senanayake")
    print("=" * 60)


if __name__ == "__main__":
    main()