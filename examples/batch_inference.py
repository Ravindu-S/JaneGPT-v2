"""
Batch inference example for JaneGPT v2 Intent Classifier.

Classifies multiple inputs efficiently.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import json
from pathlib import Path
from typing import List, Dict

import torch

from model.classifier import JaneGPTClassifier


def classify_batch(
    classifier: JaneGPTClassifier,
    texts: List[str],
    context: dict = None
) -> List[Dict]:
    """
    Classify a batch of texts.
    
    Note: Current implementation processes sequentially.
    For true batch processing with padding, see classify_batch_parallel().
    
    Args:
        classifier: Loaded JaneGPTClassifier
        texts: List of user utterances
        context: Optional shared context
        
    Returns:
        List of result dictionaries
    """
    results = []
    
    for text in texts:
        intent, confidence = classifier.predict(text, context)
        results.append({
            "text": text,
            "intent": intent,
            "confidence": round(confidence, 4),
        })
    
    return results


def classify_batch_parallel(
    classifier: JaneGPTClassifier,
    texts: List[str],
    context: dict = None
) -> List[Dict]:
    """
    Classify a batch of texts in parallel (single forward pass).
    
    More efficient for large batches on GPU.
    
    Args:
        classifier: Loaded JaneGPTClassifier
        texts: List of user utterances
        context: Optional shared context
        
    Returns:
        List of result dictionaries
    """
    if not classifier.is_ready:
        raise RuntimeError("Model not loaded")
    
    # Format and tokenize all inputs
    all_ids = []
    for text in texts:
        formatted = classifier._format_input(text, context)
        ids = classifier.tokenizer.encode(formatted).ids
        
        if len(ids) > classifier.MAX_LEN:
            ids = ids[:classifier.MAX_LEN]
        else:
            ids = ids + [classifier.PAD_ID] * (classifier.MAX_LEN - len(ids))
        
        all_ids.append(ids)
    
    # Create batch tensor
    batch_tensor = torch.tensor(all_ids, dtype=torch.long, device=classifier.device)
    
    # Single forward pass
    with torch.no_grad():
        logits, _ = classifier.model(batch_tensor)
        probs = torch.softmax(logits, dim=-1)
        confidences, predicted = torch.max(probs, dim=-1)
    
    # Build results
    results = []
    for i, text in enumerate(texts):
        idx = predicted[i].item()
        conf = confidences[i].item()
        intent = classifier.id_to_intent.get(idx, 'chat')
        
        results.append({
            "text": text,
            "intent": intent,
            "confidence": round(conf, 4),
        })
    
    return results


def main():
    # Load model
    classifier = JaneGPTClassifier()
    print(f"Model loaded: {classifier}\n")
    
    # Example batch
    commands = [
        "turn up the volume",
        "make it louder",
        "open chrome",
        "play shape of you",
        "search for python tutorials on google",
        "set brightness to 50",
        "take a screenshot",
        "set a reminder for 10 minutes",
        "mute",
        "read this for me",
        "explain what's on my screen",
        "undo that",
        "shut down",
        "hello",
        "what can you do",
        "close notepad",
        "skip to the next song",
        "dim the screen",
        "pause the music",
        "what time is it",
    ]
    
    # --- Sequential processing ---
    print("=" * 65)
    print("  Sequential Batch Processing")
    print("=" * 65)
    
    start = time.perf_counter()
    results = classify_batch(classifier, commands)
    elapsed = time.perf_counter() - start
    
    print(f"\n  {'Text':<42} {'Intent':<20} {'Conf':>6}")
    print(f"  {'-'*68}")
    
    for r in results:
        print(f"  {r['text']:<42} {r['intent']:<20} {r['confidence']:>5.1%}")
    
    print(f"\n  Processed {len(commands)} commands in {elapsed*1000:.1f}ms")
    print(f"  Average: {elapsed/len(commands)*1000:.1f}ms per command")
    
    # --- Parallel processing ---
    print(f"\n{'=' * 65}")
    print("  Parallel Batch Processing (single forward pass)")
    print("=" * 65)
    
    start = time.perf_counter()
    results_parallel = classify_batch_parallel(classifier, commands)
    elapsed_parallel = time.perf_counter() - start
    
    print(f"\n  Processed {len(commands)} commands in {elapsed_parallel*1000:.1f}ms")
    print(f"  Average: {elapsed_parallel/len(commands)*1000:.1f}ms per command")
    print(f"  Speedup: {elapsed/elapsed_parallel:.1f}x faster than sequential")
    
    # Verify both methods give same results
    match = all(
        r1['intent'] == r2['intent']
        for r1, r2 in zip(results, results_parallel)
    )
    print(f"  Results match: {'YES' if match else 'NO'}")
    
    # --- Save results to JSON ---
    output_file = Path("examples/batch_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {output_file}")
    
    # --- Batch with context ---
    print(f"\n{'=' * 65}")
    print("  Context-Aware Batch")
    print("=" * 65)
    
    # Simulate: user just adjusted volume, now giving follow-up commands
    context = {"last_intent": "volume_up"}
    follow_ups = [
        "not enough",
        "too much",
        "a bit more",
        "the other one",
        "perfect",
    ]
    
    print(f"\n  Context: last_intent = volume_up\n")
    
    ctx_results = classify_batch(classifier, follow_ups, context)
    for r in ctx_results:
        print(f"  {r['text']:<42} {r['intent']:<20} {r['confidence']:>5.1%}")


if __name__ == "__main__":
    main()