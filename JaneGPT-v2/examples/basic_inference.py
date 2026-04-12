"""
Basic inference example for JaneGPT v2 Intent Classifier.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.classifier import JaneGPTClassifier

def main():
    # Load model
    classifier = JaneGPTClassifier()
    print(f"Model loaded: {classifier}")
    print(f"Supported intents: {len(classifier.get_supported_intents())}\n")
    
    # Test commands
    test_inputs = [
        "turn up the volume",
        "make it louder",
        "set volume to 50",
        "mute",
        "turn down the brightness",
        "open chrome",
        "play shape of you on youtube",
        "search for python tutorials",
        "set a reminder for 10 minutes",
        "take a screenshot",
        "read this for me",
        "explain what's on my screen",
        "undo that",
        "shut down",
        "hello",
        "what time is it",
    ]
    
    print(f"{'Input':<45} {'Intent':<20} {'Confidence':<10}")
    print("-" * 75)
    
    for text in test_inputs:
        intent, confidence = classifier.predict(text)
        print(f"{text:<45} {intent:<20} {confidence:.1%}")
    
    # Context-aware classification
    print("\n--- Context-Aware ---")
    
    # After volume up, user says "not enough"
    intent, conf = classifier.predict(
        "not enough",
        context={"last_intent": "volume_up"}
    )
    print(f"{'not enough [after volume_up]':<45} {intent:<20} {conf:.1%}")
    
    # Top-k predictions
    print("\n--- Top-3 Predictions ---")
    results = classifier.predict_top_k("play something nice", k=3)
    for intent, conf in results:
        print(f"  {intent}: {conf:.1%}")


if __name__ == "__main__":
    main()