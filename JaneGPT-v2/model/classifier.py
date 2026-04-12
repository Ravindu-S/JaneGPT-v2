"""
JaneGPT v2 Intent Classifier — Inference Wrapper

Simple interface for intent classification.

Usage:
    from model.classifier import JaneGPTClassifier
    
    classifier = JaneGPTClassifier()
    intent, confidence = classifier.predict("turn up the volume")

Created by Ravindu Senanayake
"""

from pathlib import Path
from typing import Optional, Dict, Tuple, List

import torch

from model.architecture import JaneGPTv2Classifier, ID_TO_INTENT, INTENT_LABELS


class JaneGPTClassifier:
    """
    Ready-to-use intent classifier.
    
    Loads the trained model and tokenizer, provides simple
    predict() interface for intent classification.
    
    Args:
        model_path: Path to trained checkpoint (.pt file)
        tokenizer_path: Path to BPE tokenizer (.json file)
        device: "auto", "cuda", or "cpu"
    """
    
    MAX_LEN = 128
    PAD_ID = 0
    
    def __init__(
        self,
        model_path: str = "weights/janegpt_v2_classifier.pt",
        tokenizer_path: str = "weights/tokenizer.json",
        device: str = "auto",
    ):
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.is_ready = False
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.tokenizer = None
        self.model = None
        self.id_to_intent = ID_TO_INTENT
        
        self._load()
    
    def _load(self):
        """Load model and tokenizer."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {self.tokenizer_path}")
        
        # Load tokenizer
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        
        # Load model
        checkpoint = torch.load(
            self.model_path, map_location=self.device, weights_only=False
        )
        
        config = checkpoint.get('config', {})
        
        self.model = JaneGPTv2Classifier(
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
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        self.is_ready = True
    
    def _format_input(self, text: str, context: Optional[Dict] = None) -> str:
        """Format input for the model."""
        if context and context.get('last_intent'):
            ctx_str = f"last_action={context['last_intent']}"
        else:
            ctx_str = "none"
        
        return f"user: {text}\ncontext: {ctx_str}\njane:"
    
    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize and pad to MAX_LEN."""
        ids = self.tokenizer.encode(text).ids
        
        if len(ids) > self.MAX_LEN:
            ids = ids[:self.MAX_LEN]
        else:
            ids = ids + [self.PAD_ID] * (self.MAX_LEN - len(ids))
        
        return torch.tensor([ids], dtype=torch.long, device=self.device)
    
    def predict(
        self,
        text: str,
        context: Optional[Dict] = None
    ) -> Tuple[str, float]:
        """
        Predict intent for given text.
        
        Args:
            text: User utterance (e.g., "turn up the volume")
            context: Optional dict with 'last_intent' key
            
        Returns:
            Tuple of (intent_label, confidence)
            
        Example:
            >>> classifier.predict("open chrome")
            ('app_launch', 0.981)
        """
        if not self.is_ready:
            raise RuntimeError("Model not loaded")
        
        formatted = self._format_input(text, context)
        input_ids = self._tokenize(formatted)
        
        predicted_idx, confidence = self.model.predict(input_ids)
        intent = self.id_to_intent.get(predicted_idx, 'chat')
        
        return intent, confidence
    
    def predict_top_k(
        self,
        text: str,
        context: Optional[Dict] = None,
        k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Get top-k predictions with confidences.
        
        Args:
            text: User utterance
            context: Optional context dict
            k: Number of top predictions to return
            
        Returns:
            List of (intent_label, confidence) tuples
            
        Example:
            >>> classifier.predict_top_k("play something", k=3)
            [('media_play', 0.85), ('browser_search', 0.08), ('chat', 0.03)]
        """
        if not self.is_ready:
            raise RuntimeError("Model not loaded")
        
        formatted = self._format_input(text, context)
        input_ids = self._tokenize(formatted)
        
        with torch.no_grad():
            logits, _ = self.model(input_ids)
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_indices = probs.topk(k, dim=-1)
            
            return [
                (self.id_to_intent.get(idx.item(), 'chat'), prob.item())
                for prob, idx in zip(top_probs[0], top_indices[0])
            ]
    
    @staticmethod
    def get_supported_intents() -> List[str]:
        """Get list of all supported intent labels."""
        return INTENT_LABELS.copy()
    
    def __repr__(self):
        return (
            f"JaneGPTClassifier("
            f"ready={self.is_ready}, "
            f"device={self.device}, "
            f"intents={len(INTENT_LABELS)})"
        )