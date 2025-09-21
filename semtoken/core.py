"""
Core SemToken implementation with the main semantic-aware tokenization pipeline.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from transformers import AutoTokenizer

from .semantic_encoder import SemanticEncoder
from .clustering import SemanticCluster
from .granularity import GranularityAssigner
from .budget import BudgetAllocator
from .utils import TokenMerger, SemanticAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class SemTokenConfig:
    """Configuration for SemToken pipeline."""
    
    # Semantic encoder settings
    encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    context_window: int = 5
    embedding_dim: int = 384
    
    # Clustering parameters
    similarity_threshold: float = 0.7
    min_cluster_size: int = 2
    max_cluster_size: int = 10
    
    # Granularity assignment
    entropy_threshold: float = 0.5
    density_window: int = 10
    
    # Budget allocation
    compression_ratio: float = 0.5
    max_budget: Optional[int] = None
    priority_boost: float = 1.2
    
    # Implementation settings
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cache_embeddings: bool = True
    parallel_processing: bool = True


class SemToken:
    """
    Main SemToken class implementing semantic-aware tokenization.
    
    SemToken operates in three main stages:
    1. Semantic Embedding: Extract contextual embeddings using lightweight encoders
    2. Local Clustering: Merge semantically equivalent tokens based on similarity
    3. Granularity Assignment: Allocate variable token granularity based on semantic density
    """
    
    def __init__(self, config: Optional[SemTokenConfig] = None, tokenizer=None):
        """
        Initialize SemToken with configuration.
        
        Args:
            config: SemTokenConfig instance with pipeline parameters
            tokenizer: Base tokenizer (e.g., BPE, WordPiece) to build upon
        """
        self.config = config or SemTokenConfig()
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained("gpt2")
        
        # Initialize pipeline components
        self.semantic_encoder = SemanticEncoder(
            model_name=self.config.encoder_model,
            context_window=self.config.context_window,
            device=self.config.device
        )
        
        self.clusterer = SemanticCluster(
            similarity_threshold=self.config.similarity_threshold,
            min_cluster_size=self.config.min_cluster_size,
            max_cluster_size=self.config.max_cluster_size
        )
        
        self.granularity_assigner = GranularityAssigner(
            entropy_threshold=self.config.entropy_threshold,
            density_window=self.config.density_window
        )
        
        self.budget_allocator = BudgetAllocator(
            compression_ratio=self.config.compression_ratio,
            max_budget=self.config.max_budget,
            priority_boost=self.config.priority_boost
        )
        
        self.token_merger = TokenMerger(self.tokenizer)
        self.semantic_analyzer = SemanticAnalyzer()
        
        # Cache for embeddings if enabled
        self.embedding_cache = {} if self.config.cache_embeddings else None
        
        logger.info(f"SemToken initialized with compression ratio: {self.config.compression_ratio}")
    
    def tokenize(self, text: str, query: Optional[str] = None) -> Dict:
        """
        Main tokenization pipeline that applies semantic-aware compression.
        
        Args:
            text: Input text to tokenize
            query: Optional query for query-aware compression
            
        Returns:
            Dictionary containing:
            - compressed_tokens: List of compressed token IDs
            - token_spans: Mapping from compressed tokens to original spans
            - compression_stats: Statistics about the compression
            - semantic_info: Semantic analysis results
        """
        logger.info(f"Processing text of length {len(text)}")
        
        # Step 1: Initial tokenization with base tokenizer
        initial_tokens = self.tokenizer.encode(text, return_tensors="pt")
        original_text_tokens = self.tokenizer.convert_ids_to_tokens(initial_tokens[0])
        
        logger.debug(f"Initial tokens: {len(original_text_tokens)}")
        
        # Step 2: Extract semantic embeddings
        embeddings = self._extract_embeddings(original_text_tokens, text)
        
        # Step 3: Perform local clustering to identify semantically similar spans
        clusters = self.clusterer.cluster_tokens(original_text_tokens, embeddings)
        
        # Step 4: Assign granularity based on semantic density
        granularity_assignments = self.granularity_assigner.assign_granularity(
            clusters, embeddings
        )
        
        # Step 5: Budget-aware token allocation
        selected_spans = self.budget_allocator.allocate_budget(
            clusters, granularity_assignments, query_embedding=self._get_query_embedding(query)
        )
        
        # Step 6: Merge tokens according to selection
        compressed_result = self.token_merger.merge_tokens(
            original_text_tokens, selected_spans, initial_tokens[0]
        )
        
        # Step 7: Compile results and statistics
        result = self._compile_results(
            compressed_result, original_text_tokens, clusters, granularity_assignments
        )
        
        logger.info(f"Compression: {len(original_text_tokens)} -> {len(result['compressed_tokens'])} "
                   f"({result['compression_stats']['compression_ratio']:.2%})")
        
        return result
    
    def _extract_embeddings(self, tokens: List[str], text: str) -> torch.Tensor:
        """Extract contextual embeddings for tokens."""
        cache_key = hash(text) if self.embedding_cache is not None else None
        
        if cache_key and cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        embeddings = self.semantic_encoder.encode_tokens(tokens, text)
        
        if cache_key:
            self.embedding_cache[cache_key] = embeddings
            
        return embeddings
    
    def _get_query_embedding(self, query: Optional[str]) -> Optional[torch.Tensor]:
        """Get embedding for query if provided."""
        if query is None:
            return None
        return self.semantic_encoder.encode_text(query)
    
    def _compile_results(self, compressed_result: Dict, original_tokens: List[str], 
                        clusters: List, granularity_assignments: Dict) -> Dict:
        """Compile final results with statistics and metadata."""
        
        original_count = len(original_tokens)
        compressed_count = len(compressed_result['token_ids'])
        compression_ratio = compressed_count / original_count
        
        # Calculate semantic statistics
        semantic_stats = self.semantic_analyzer.analyze_compression(
            original_tokens, compressed_result, clusters
        )
        
        return {
            'compressed_tokens': compressed_result['token_ids'],
            'token_spans': compressed_result['span_mapping'],
            'compression_stats': {
                'original_count': original_count,
                'compressed_count': compressed_count,
                'compression_ratio': compression_ratio,
                'tokens_saved': original_count - compressed_count,
                'theoretical_speedup': 1.0 / compression_ratio,
                'memory_reduction': 1.0 - compression_ratio
            },
            'semantic_info': {
                'clusters_formed': len(clusters),
                'granularity_distribution': granularity_assignments,
                'semantic_stats': semantic_stats
            },
            'metadata': {
                'config': self.config,
                'processing_time': compressed_result.get('processing_time', 0)
            }
        }
    
    def decode(self, token_ids: List[int], span_mapping: Dict) -> str:
        """
        Decode compressed tokens back to text.
        
        Args:
            token_ids: Compressed token IDs
            span_mapping: Mapping from compressed tokens to original spans
            
        Returns:
            Decoded text string
        """
        return self.token_merger.decode_tokens(token_ids, span_mapping)
    
    def get_compression_stats(self) -> Dict:
        """Get statistics about recent compression operations."""
        return {
            'cache_size': len(self.embedding_cache) if self.embedding_cache else 0,
            'config': self.config.__dict__,
            'model_info': {
                'encoder_model': self.config.encoder_model,
                'device': self.config.device,
                'embedding_dim': self.config.embedding_dim
            }
        }
    
    def clear_cache(self):
        """Clear embedding cache to free memory."""
        if self.embedding_cache:
            self.embedding_cache.clear()
            logger.info("Embedding cache cleared")
    
    def update_config(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config: {key} = {value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")
    
    def batch_process(self, texts: List[str], queries: Optional[List[str]] = None) -> List[Dict]:
        """
        Process multiple texts in batch for efficiency.
        
        Args:
            texts: List of input texts
            queries: Optional list of queries for query-aware compression
            
        Returns:
            List of compression results
        """
        if queries and len(queries) != len(texts):
            raise ValueError("Number of queries must match number of texts")
        
        results = []
        for i, text in enumerate(texts):
            query = queries[i] if queries else None
            result = self.tokenize(text, query)
            results.append(result)
            
        return results


# Algorithm 1 implementation as a standalone function
def semtoken_algorithm(tokens: List[str], encoder, similarity_threshold: float = 0.7, 
                      entropy_threshold: float = 0.5, budget: int = None) -> List[str]:
    """
    Direct implementation of Algorithm 1 from the paper.
    
    Args:
        tokens: Input token sequence
        encoder: Semantic encoder function
        similarity_threshold: Threshold for token similarity (τ)
        entropy_threshold: Threshold for semantic entropy (δ)  
        budget: Maximum number of output tokens (B)
        
    Returns:
        Compressed token sequence
    """
    n = len(tokens)
    if budget is None:
        budget = int(n * 0.5)  # Default 50% compression
    
    # Step 1: Semantic Fingerprint Extraction
    embeddings = []
    for i in range(n):
        context_start = max(0, i - 2)  # k=2 for context window
        context_end = min(n, i + 3)
        context_tokens = tokens[context_start:context_end]
        embedding = encoder.encode_tokens(context_tokens, "")
        embeddings.append(embedding)
    
    # Step 2: Span Formation via Local Similarity
    spans = []
    t = 0
    while t < n:
        current_span = [t]
        
        for j in range(t + 1, n):
            # Calculate cosine similarity
            similarity = torch.cosine_similarity(
                embeddings[t].unsqueeze(0), 
                embeddings[j].unsqueeze(0)
            ).item()
            
            if similarity > similarity_threshold:
                current_span.append(j)
            else:
                break
        
        spans.append(current_span)
        t = current_span[-1] + 1
    
    # Step 3: Semantic Entropy Scoring
    span_entropies = []
    for span_indices in spans:
        span_embeddings = [embeddings[i] for i in span_indices]
        if len(span_embeddings) > 1:
            # Calculate covariance matrix trace as entropy measure
            stacked = torch.stack(span_embeddings)
            cov_matrix = torch.cov(stacked.T)
            entropy = torch.trace(cov_matrix).item()
        else:
            entropy = 1.0  # Single token has maximum entropy
        span_entropies.append(entropy)
    
    # Step 4: Entropy-Guided Selection under Budget
    # Sort spans by entropy (descending) and select top-B
    span_entropy_pairs = list(zip(spans, span_entropies))
    span_entropy_pairs.sort(key=lambda x: x[1], reverse=True)
    
    selected_spans = [span for span, _ in span_entropy_pairs[:budget]]
    
    # Merge tokens in selected spans
    compressed_tokens = []
    for span_indices in selected_spans:
        if len(span_indices) == 1:
            compressed_tokens.append(tokens[span_indices[0]])
        else:
            # Merge multiple tokens into one
            merged_token = "".join(tokens[i] for i in span_indices)
            compressed_tokens.append(merged_token)
    
    return compressed_tokens
