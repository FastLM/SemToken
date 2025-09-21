"""
Semantic encoder module for extracting contextual embeddings from token sequences.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Union
import logging
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import hashlib
from functools import lru_cache

logger = logging.getLogger(__name__)


class SemanticEncoder:
    """
    Lightweight semantic encoder for extracting contextual embeddings.
    
    Supports multiple encoder backends including:
    - Sentence-BERT models (SimCSE, all-MiniLM, etc.)
    - Distilled BERT models
    - Custom fine-tuned encoders
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 context_window: int = 5, device: str = "cpu", 
                 max_seq_length: int = 512, batch_size: int = 32):
        """
        Initialize semantic encoder.
        
        Args:
            model_name: HuggingFace model name or path
            context_window: Size of context window around each token
            device: Device to run inference on
            max_seq_length: Maximum sequence length for the model
            batch_size: Batch size for processing
        """
        self.model_name = model_name
        self.context_window = context_window
        self.device = torch.device(device)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        
        # Initialize model based on type
        self._load_model()
        
        # Cache for frequently accessed embeddings
        self._embedding_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.info(f"SemanticEncoder initialized with {model_name} on {device}")
    
    def _load_model(self):
        """Load the appropriate model based on model_name."""
        try:
            if "sentence-transformers" in self.model_name or self.model_name.startswith("all-"):
                # Use SentenceTransformer for sentence embedding models
                self.model = SentenceTransformer(self.model_name, device=self.device)
                self.tokenizer = self.model.tokenizer
                self.model_type = "sentence_transformer"
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
            else:
                # Use standard transformers for BERT-like models
                self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model_type = "transformer"
                self.embedding_dim = self.model.config.hidden_size
                
            self.model.eval()  # Set to evaluation mode
            logger.info(f"Loaded {self.model_type} model with embedding dim: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode a single text string to embedding.
        
        Args:
            text: Input text string
            
        Returns:
            Embedding tensor of shape [embedding_dim]
        """
        # Check cache first
        cache_key = self._get_cache_key(text)
        if cache_key in self._embedding_cache:
            self._cache_hits += 1
            return self._embedding_cache[cache_key]
        
        self._cache_misses += 1
        
        with torch.no_grad():
            if self.model_type == "sentence_transformer":
                embedding = self.model.encode(text, convert_to_tensor=True, device=self.device)
            else:
                # Standard transformer encoding
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                                      max_length=self.max_seq_length, padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                # Use mean pooling of last hidden states
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        
        # Cache the result
        self._embedding_cache[cache_key] = embedding
        return embedding
    
    def encode_tokens(self, tokens: List[str], original_text: str = "") -> torch.Tensor:
        """
        Encode token sequence with contextual information.
        
        Args:
            tokens: List of token strings
            original_text: Original text for additional context
            
        Returns:
            Embedding tensor of shape [num_tokens, embedding_dim]
        """
        embeddings = []
        
        for i, token in enumerate(tokens):
            # Create contextual window around current token
            start_idx = max(0, i - self.context_window)
            end_idx = min(len(tokens), i + self.context_window + 1)
            
            context_tokens = tokens[start_idx:end_idx]
            context_text = self._reconstruct_text(context_tokens, token, i - start_idx)
            
            # Get embedding for contextualized token
            embedding = self.encode_text(context_text)
            embeddings.append(embedding)
        
        return torch.stack(embeddings)
    
    def encode_spans(self, token_spans: List[List[str]]) -> torch.Tensor:
        """
        Encode multiple token spans in batch.
        
        Args:
            token_spans: List of token span lists
            
        Returns:
            Embedding tensor of shape [num_spans, embedding_dim]
        """
        span_texts = [self._reconstruct_text(span) for span in token_spans]
        return self.encode_batch(span_texts)
    
    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """
        Encode multiple texts in batch for efficiency.
        
        Args:
            texts: List of text strings
            
        Returns:
            Embedding tensor of shape [num_texts, embedding_dim]
        """
        if not texts:
            return torch.empty(0, self.embedding_dim, device=self.device)
        
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            with torch.no_grad():
                if self.model_type == "sentence_transformer":
                    batch_embeddings = self.model.encode(
                        batch_texts, convert_to_tensor=True, device=self.device,
                        batch_size=len(batch_texts)
                    )
                else:
                    # Standard transformer batch encoding
                    inputs = self.tokenizer(
                        batch_texts, return_tensors="pt", truncation=True,
                        max_length=self.max_seq_length, padding=True
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    outputs = self.model(**inputs)
                    batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            
            embeddings.append(batch_embeddings)
        
        return torch.cat(embeddings, dim=0)
    
    def compute_semantic_entropy(self, embeddings: torch.Tensor) -> float:
        """
        Compute semantic entropy for a set of embeddings.
        
        Uses the trace of covariance matrix as entropy measure:
        H(T) = Tr(Cov({f_θ(x_i) | x_i ∈ T}))
        
        Args:
            embeddings: Tensor of shape [num_embeddings, embedding_dim]
            
        Returns:
            Semantic entropy score
        """
        if embeddings.size(0) <= 1:
            return 1.0  # Maximum entropy for single embedding
        
        # Compute covariance matrix
        centered = embeddings - embeddings.mean(dim=0, keepdim=True)
        cov_matrix = torch.mm(centered.t(), centered) / (embeddings.size(0) - 1)
        
        # Return trace as entropy measure
        entropy = torch.trace(cov_matrix).item()
        
        # Normalize by embedding dimension for comparable scores
        return entropy / self.embedding_dim
    
    def compute_similarity_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise cosine similarity matrix for embeddings.
        
        Args:
            embeddings: Tensor of shape [num_embeddings, embedding_dim]
            
        Returns:
            Similarity matrix of shape [num_embeddings, num_embeddings]
        """
        # Normalize embeddings
        normalized = nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Compute cosine similarity matrix
        similarity_matrix = torch.mm(normalized, normalized.t())
        
        return similarity_matrix
    
    def find_similar_tokens(self, embeddings: torch.Tensor, 
                           threshold: float = 0.7) -> List[List[int]]:
        """
        Find groups of similar tokens based on embedding similarity.
        
        Args:
            embeddings: Token embeddings tensor
            threshold: Similarity threshold for grouping
            
        Returns:
            List of token index groups
        """
        similarity_matrix = self.compute_similarity_matrix(embeddings)
        n = similarity_matrix.size(0)
        
        visited = set()
        groups = []
        
        for i in range(n):
            if i in visited:
                continue
                
            # Find all tokens similar to token i
            similar_indices = [i]
            for j in range(i + 1, n):
                if j not in visited and similarity_matrix[i, j] > threshold:
                    similar_indices.append(j)
                    visited.add(j)
            
            if len(similar_indices) > 1:
                groups.append(similar_indices)
                visited.update(similar_indices)
        
        return groups
    
    def _reconstruct_text(self, tokens: List[str], focus_token: str = None, 
                         focus_idx: int = None) -> str:
        """
        Reconstruct text from tokens with proper spacing.
        
        Args:
            tokens: List of token strings
            focus_token: Token to emphasize (optional)
            focus_idx: Index of focus token (optional)
            
        Returns:
            Reconstructed text string
        """
        if not tokens:
            return ""
        
        # Simple reconstruction - can be improved with better detokenization
        text = ""
        for i, token in enumerate(tokens):
            # Handle special tokens and spacing
            if token.startswith("##"):  # WordPiece continuation
                text += token[2:]
            elif token.startswith("Ġ"):  # GPT-style space token
                text += " " + token[1:]
            elif i == 0:
                text += token
            else:
                # Add space before token unless it's punctuation
                if not token in ".,!?;:":
                    text += " "
                text += token
        
        return text.strip()
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_size": len(self._embedding_cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }
    
    def clear_cache(self):
        """Clear embedding cache."""
        self._embedding_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Embedding cache cleared")
    
    def save_embeddings(self, filepath: str, embeddings: torch.Tensor, metadata: Dict = None):
        """Save embeddings to file for later use."""
        torch.save({
            "embeddings": embeddings,
            "metadata": metadata or {},
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim
        }, filepath)
        logger.info(f"Embeddings saved to {filepath}")
    
    def load_embeddings(self, filepath: str) -> Tuple[torch.Tensor, Dict]:
        """Load embeddings from file."""
        data = torch.load(filepath, map_location=self.device)
        logger.info(f"Embeddings loaded from {filepath}")
        return data["embeddings"], data.get("metadata", {})


class FastSemanticEncoder(SemanticEncoder):
    """
    Optimized version of SemanticEncoder for faster inference.
    
    Uses techniques like:
    - Quantized models
    - Reduced precision
    - Aggressive caching
    - Parallel processing
    """
    
    def __init__(self, *args, use_quantization: bool = True, 
                 use_half_precision: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.use_quantization = use_quantization
        self.use_half_precision = use_half_precision
        
        if use_half_precision and self.device.type == "cuda":
            self.model = self.model.half()
            logger.info("Using half precision for faster inference")
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Optimized text encoding with reduced precision."""
        embedding = super().encode_text(text)
        
        if self.use_half_precision and self.device.type == "cuda":
            embedding = embedding.half()
            
        return embedding


# Utility functions for semantic fingerprinting
def compute_semantic_fingerprint(embeddings: torch.Tensor, 
                                method: str = "mean") -> torch.Tensor:
    """
    Compute semantic fingerprint for a set of embeddings.
    
    Args:
        embeddings: Tensor of shape [num_embeddings, embedding_dim]
        method: Fingerprinting method ("mean", "max", "attention")
        
    Returns:
        Fingerprint tensor of shape [embedding_dim]
    """
    if method == "mean":
        return embeddings.mean(dim=0)
    elif method == "max":
        return embeddings.max(dim=0)[0]
    elif method == "attention":
        # Use attention-weighted pooling
        attention_weights = torch.softmax(embeddings.sum(dim=1), dim=0)
        return (embeddings * attention_weights.unsqueeze(1)).sum(dim=0)
    else:
        raise ValueError(f"Unknown fingerprinting method: {method}")


def stride_based_fingerprinting(tokens: List[str], encoder: SemanticEncoder, 
                               stride: int = 4) -> torch.Tensor:
    """
    Efficient stride-based fingerprinting for parallelism.
    
    Args:
        tokens: List of token strings
        encoder: SemanticEncoder instance
        stride: Stride size for processing
        
    Returns:
        Fingerprints tensor of shape [num_strides, embedding_dim]
    """
    fingerprints = []
    
    for i in range(0, len(tokens), stride):
        end_idx = min(i + stride, len(tokens))
        token_chunk = tokens[i:end_idx]
        
        # Get embeddings for chunk
        embeddings = encoder.encode_tokens(token_chunk)
        
        # Compute fingerprint for chunk
        fingerprint = compute_semantic_fingerprint(embeddings)
        fingerprints.append(fingerprint)
    
    return torch.stack(fingerprints)
