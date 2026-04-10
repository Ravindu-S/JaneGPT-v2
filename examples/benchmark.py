"""
Speed benchmark for JaneGPT v2.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
from model.classifier import JaneGPTClassifier

def main():
    classifier = JaneGPTClassifier()
    
    test_inputs = [
        "turn up the volume",
        "open chrome",
        "play some music",
        "set brightness to 50",
        "search for cats",
        "take a screenshot",
        "hello",
        "undo that",
    ]
    
    # Warmup
    for text in test_inputs:
        classifier.predict(text)
    
    # Benchmark
    iterations = 100
    start = time.perf_counter()
    
    for _ in range(iterations):
        for text in test_inputs:
            classifier.predict(text)
    
    elapsed = time.perf_counter() - start
    total_predictions = iterations * len(test_inputs)
    
    print(f"Device: {classifier.device}")
    print(f"Total predictions: {total_predictions}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Average per prediction: {elapsed/total_predictions*1000:.2f}ms")
    print(f"Predictions per second: {total_predictions/elapsed:.0f}")


if __name__ == "__main__":
    main()