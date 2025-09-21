"""
Utility functions and helper classes for SemToken.
"""

import torch
import numpy as np
import json
import pickle
import time
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import logging
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class TokenSpan:
    """Represents a span of tokens with metadata."""
    start_idx: int
    end_idx: int
    tokens: List[str]
    original_ids: List[int]
    merged_id: Optional[int] = None
    compression_ratio: float = 1.0
    semantic_score: float = 0.0


class TokenMerger:
    """
    Handles merging and reconstruction of compressed tokens.
    
    Maintains mappings between original tokens and compressed representations
    while preserving decoding capability.
    """
    
    def __init__(self, base_tokenizer):
        """
        Initialize token merger.
        
        Args:
            base_tokenizer: Base tokenizer (e.g., GPT-2, BERT tokenizer)
        """
        self.base_tokenizer = base_tokenizer
        self.vocab_size = len(base_tokenizer.get_vocab())
        
        # Extended vocabulary for merged tokens
        self.merged_vocab = {}  # merged_token_id -> original_token_ids
        self.reverse_merged_vocab = {}  # tuple(original_ids) -> merged_token_id
        
        # Special tokens for merged content
        self.merge_token_start = self.vocab_size
        self.next_merge_id = self.merge_token_start
        
        logger.info(f"TokenMerger initialized with base vocab size: {self.vocab_size}")
    
    def merge_tokens(self, original_tokens: List[str], selected_spans: List[int],
                    original_ids: torch.Tensor) -> Dict:
        """
        Merge tokens according to selected spans.
        
        Args:
            original_tokens: List of original token strings
            selected_spans: List of cluster IDs that were selected
            original_ids: Original token IDs tensor
            
        Returns:
            Dictionary with merged tokens and metadata
        """
        start_time = time.time()
        
        # Create token spans for merging
        token_spans = self._create_token_spans(
            original_tokens, selected_spans, original_ids
        )
        
        # Perform actual merging
        merged_ids = []
        span_mapping = {}
        compression_stats = {"spans_merged": 0, "tokens_saved": 0}
        
        current_pos = 0
        for span in token_spans:
            if span.compression_ratio < 1.0:  # This span was compressed
                # Create or reuse merged token ID
                merged_id = self._get_merged_token_id(span.original_ids)
                merged_ids.append(merged_id)
                
                # Update mappings
                span_mapping[merged_id] = {
                    "original_tokens": span.tokens,
                    "original_ids": span.original_ids,
                    "start_pos": current_pos,
                    "end_pos": current_pos + len(span.tokens) - 1,
                    "compression_ratio": span.compression_ratio
                }
                
                compression_stats["spans_merged"] += 1
                compression_stats["tokens_saved"] += len(span.tokens) - 1
                
            else:  # Span was not compressed
                merged_ids.extend(span.original_ids)
            
            current_pos += len(span.tokens)
        
        processing_time = time.time() - start_time
        
        return {
            "token_ids": merged_ids,
            "span_mapping": span_mapping,
            "compression_stats": compression_stats,
            "processing_time": processing_time,
            "original_length": len(original_tokens),
            "compressed_length": len(merged_ids)
        }
    
    def decode_tokens(self, token_ids: List[int], span_mapping: Dict) -> str:
        """
        Decode compressed tokens back to text.
        
        Args:
            token_ids: List of token IDs (including merged tokens)
            span_mapping: Mapping from merged tokens to original spans
            
        Returns:
            Decoded text string
        """
        decoded_tokens = []
        
        for token_id in token_ids:
            if token_id in span_mapping:
                # This is a merged token - use original tokens
                original_tokens = span_mapping[token_id]["original_tokens"]
                decoded_tokens.extend(original_tokens)
            else:
                # Regular token - decode normally
                if token_id < self.vocab_size:
                    decoded_token = self.base_tokenizer.convert_ids_to_tokens([token_id])[0]
                    decoded_tokens.append(decoded_token)
        
        # Reconstruct text from tokens
        return self._reconstruct_text_from_tokens(decoded_tokens)
    
    def _create_token_spans(self, tokens: List[str], selected_spans: List[int],
                           original_ids: torch.Tensor) -> List[TokenSpan]:
        """Create TokenSpan objects for merging."""
        
        # For now, create simple spans - in practice would use cluster information
        spans = []
        current_start = 0
        
        # Group consecutive tokens into spans
        i = 0
        while i < len(tokens):
            span_length = min(3, len(tokens) - i)  # Max span length of 3
            
            span = TokenSpan(
                start_idx=current_start,
                end_idx=current_start + span_length - 1,
                tokens=tokens[i:i + span_length],
                original_ids=original_ids[i:i + span_length].tolist(),
                compression_ratio=0.5 if span_length > 1 else 1.0  # Compress multi-token spans
            )
            
            spans.append(span)
            current_start += span_length
            i += span_length
        
        return spans
    
    def _get_merged_token_id(self, original_ids: List[int]) -> int:
        """Get or create merged token ID for a sequence of original IDs."""
        
        key = tuple(original_ids)
        
        if key in self.reverse_merged_vocab:
            return self.reverse_merged_vocab[key]
        
        # Create new merged token ID
        merged_id = self.next_merge_id
        self.next_merge_id += 1
        
        # Update vocabularies
        self.merged_vocab[merged_id] = original_ids
        self.reverse_merged_vocab[key] = merged_id
        
        return merged_id
    
    def _reconstruct_text_from_tokens(self, tokens: List[str]) -> str:
        """Reconstruct text from token list."""
        
        if not tokens:
            return ""
        
        # Simple reconstruction - can be improved
        text = ""
        for i, token in enumerate(tokens):
            # Handle different tokenizer formats
            if token.startswith("##"):  # WordPiece continuation
                text += token[2:]
            elif token.startswith("Ġ"):  # GPT-style space token
                if i > 0:
                    text += " "
                text += token[1:]
            elif i == 0:
                text += token
            else:
                # Add space before token unless it's punctuation
                if not token in ".,!?;:":
                    text += " "
                text += token
        
        return text.strip()
    
    def save_vocabulary(self, filepath: str):
        """Save merged vocabulary to file."""
        vocab_data = {
            "merged_vocab": self.merged_vocab,
            "reverse_merged_vocab": {str(k): v for k, v in self.reverse_merged_vocab.items()},
            "next_merge_id": self.next_merge_id,
            "vocab_size": self.vocab_size
        }
        
        with open(filepath, 'w') as f:
            json.dump(vocab_data, f, indent=2)
        
        logger.info(f"Merged vocabulary saved to {filepath}")
    
    def load_vocabulary(self, filepath: str):
        """Load merged vocabulary from file."""
        with open(filepath, 'r') as f:
            vocab_data = json.load(f)
        
        self.merged_vocab = {int(k): v for k, v in vocab_data["merged_vocab"].items()}
        self.reverse_merged_vocab = {
            tuple(eval(k)): v for k, v in vocab_data["reverse_merged_vocab"].items()
        }
        self.next_merge_id = vocab_data["next_merge_id"]
        
        logger.info(f"Merged vocabulary loaded from {filepath}")


class SemanticAnalyzer:
    """
    Analyzes semantic properties of text and compression results.
    
    Provides insights into compression quality and semantic preservation.
    """
    
    def __init__(self):
        """Initialize semantic analyzer."""
        self.analysis_cache = {}
    
    def analyze_compression(self, original_tokens: List[str], 
                          compressed_result: Dict, 
                          clusters: List) -> Dict:
        """
        Analyze the quality of semantic compression.
        
        Args:
            original_tokens: Original token sequence
            compressed_result: Result from token merging
            clusters: Semantic clusters used
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {}
        
        # Basic compression metrics
        original_length = len(original_tokens)
        compressed_length = compressed_result["compressed_length"]
        
        analysis["compression_ratio"] = compressed_length / original_length
        analysis["tokens_saved"] = original_length - compressed_length
        analysis["compression_efficiency"] = 1.0 - analysis["compression_ratio"]
        
        # Semantic preservation metrics
        analysis["semantic_preservation"] = self._estimate_semantic_preservation(
            original_tokens, compressed_result, clusters
        )
        
        # Information density analysis
        analysis["information_density"] = self._analyze_information_density(
            original_tokens, compressed_result
        )
        
        # Cluster quality metrics
        analysis["cluster_quality"] = self._analyze_cluster_quality(clusters)
        
        # Compression uniformity
        analysis["compression_uniformity"] = self._analyze_compression_uniformity(
            compressed_result
        )
        
        return analysis
    
    def _estimate_semantic_preservation(self, original_tokens: List[str],
                                       compressed_result: Dict,
                                       clusters: List) -> float:
        """Estimate how well semantic meaning is preserved."""
        
        if not clusters:
            return 0.5  # Default score
        
        # Use cluster coherence as proxy for semantic preservation
        coherence_scores = [getattr(cluster, 'coherence_score', 0.5) for cluster in clusters]
        avg_coherence = np.mean(coherence_scores)
        
        # Adjust based on compression ratio
        compression_ratio = compressed_result["compressed_length"] / len(original_tokens)
        
        # Higher compression with high coherence = good semantic preservation
        preservation_score = avg_coherence * (1.0 + (1.0 - compression_ratio) * 0.5)
        
        return min(1.0, preservation_score)
    
    def _analyze_information_density(self, original_tokens: List[str],
                                    compressed_result: Dict) -> Dict:
        """Analyze information density of compression."""
        
        # Calculate tokens per unit of information
        original_length = len(original_tokens)
        compressed_length = compressed_result["compressed_length"]
        
        if compressed_length == 0:
            return {"density_ratio": 0, "efficiency": 0}
        
        density_ratio = original_length / compressed_length
        efficiency = (density_ratio - 1.0) / density_ratio if density_ratio > 1 else 0
        
        return {
            "density_ratio": density_ratio,
            "efficiency": efficiency,
            "information_per_token": density_ratio
        }
    
    def _analyze_cluster_quality(self, clusters: List) -> Dict:
        """Analyze the quality of semantic clusters."""
        
        if not clusters:
            return {"avg_coherence": 0, "size_distribution": {}, "quality_score": 0}
        
        coherence_scores = [getattr(cluster, 'coherence_score', 0.5) for cluster in clusters]
        cluster_sizes = [getattr(cluster, 'size', 1) for cluster in clusters]
        
        # Size distribution
        size_counts = {}
        for size in cluster_sizes:
            size_counts[size] = size_counts.get(size, 0) + 1
        
        # Quality metrics
        avg_coherence = np.mean(coherence_scores)
        coherence_std = np.std(coherence_scores)
        avg_size = np.mean(cluster_sizes)
        
        # Overall quality score
        quality_score = avg_coherence * (1.0 - coherence_std * 0.5)
        
        return {
            "avg_coherence": avg_coherence,
            "coherence_std": coherence_std,
            "avg_cluster_size": avg_size,
            "size_distribution": size_counts,
            "quality_score": quality_score
        }
    
    def _analyze_compression_uniformity(self, compressed_result: Dict) -> Dict:
        """Analyze how uniformly compression is applied."""
        
        span_mapping = compressed_result.get("span_mapping", {})
        
        if not span_mapping:
            return {"uniformity_score": 1.0, "compression_variance": 0}
        
        # Analyze compression ratios across spans
        compression_ratios = [
            span_info["compression_ratio"] 
            for span_info in span_mapping.values()
        ]
        
        if not compression_ratios:
            return {"uniformity_score": 1.0, "compression_variance": 0}
        
        variance = np.var(compression_ratios)
        uniformity_score = 1.0 / (1.0 + variance)  # Higher variance = lower uniformity
        
        return {
            "uniformity_score": uniformity_score,
            "compression_variance": variance,
            "compression_ratios": compression_ratios
        }


class PerformanceProfiler:
    """
    Profiles performance of SemToken operations.
    
    Tracks timing, memory usage, and efficiency metrics.
    """
    
    def __init__(self):
        """Initialize performance profiler."""
        self.profiles = {}
        self.current_profile = None
    
    def start_profile(self, operation_name: str):
        """Start profiling an operation."""
        self.current_profile = {
            "operation": operation_name,
            "start_time": time.time(),
            "start_memory": self._get_memory_usage(),
            "stages": []
        }
    
    def mark_stage(self, stage_name: str):
        """Mark a stage within the current operation."""
        if self.current_profile is None:
            return
        
        current_time = time.time()
        stage_info = {
            "name": stage_name,
            "timestamp": current_time,
            "elapsed": current_time - self.current_profile["start_time"],
            "memory": self._get_memory_usage()
        }
        
        self.current_profile["stages"].append(stage_info)
    
    def end_profile(self) -> Dict:
        """End current profile and return results."""
        if self.current_profile is None:
            return {}
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        profile_result = {
            "operation": self.current_profile["operation"],
            "total_time": end_time - self.current_profile["start_time"],
            "memory_delta": end_memory - self.current_profile["start_memory"],
            "stages": self.current_profile["stages"],
            "peak_memory": max(
                stage["memory"] for stage in self.current_profile["stages"]
            ) if self.current_profile["stages"] else end_memory
        }
        
        # Store in profiles
        op_name = self.current_profile["operation"]
        if op_name not in self.profiles:
            self.profiles[op_name] = []
        self.profiles[op_name].append(profile_result)
        
        self.current_profile = None
        return profile_result
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0  # psutil not available
    
    def get_performance_summary(self) -> Dict:
        """Get summary of all profiled operations."""
        summary = {}
        
        for operation, profiles in self.profiles.items():
            if not profiles:
                continue
            
            times = [p["total_time"] for p in profiles]
            memories = [p["memory_delta"] for p in profiles]
            
            summary[operation] = {
                "count": len(profiles),
                "avg_time": np.mean(times),
                "std_time": np.std(times),
                "min_time": np.min(times),
                "max_time": np.max(times),
                "avg_memory": np.mean(memories),
                "total_time": np.sum(times)
            }
        
        return summary


# Utility functions
def compute_compression_metrics(original_length: int, compressed_length: int) -> Dict:
    """Compute standard compression metrics."""
    
    if original_length == 0:
        return {"compression_ratio": 0, "space_savings": 0, "compression_factor": 0}
    
    compression_ratio = compressed_length / original_length
    space_savings = 1.0 - compression_ratio
    compression_factor = original_length / compressed_length if compressed_length > 0 else float('inf')
    
    return {
        "compression_ratio": compression_ratio,
        "space_savings": space_savings,
        "compression_factor": compression_factor,
        "tokens_saved": original_length - compressed_length
    }


def estimate_memory_savings(original_tokens: int, compressed_tokens: int,
                           embedding_dim: int = 768, precision_bytes: int = 2) -> Dict:
    """Estimate memory savings from token compression."""
    
    # Calculate KV cache memory usage
    original_kv_memory = original_tokens * embedding_dim * 2 * precision_bytes  # K and V
    compressed_kv_memory = compressed_tokens * embedding_dim * 2 * precision_bytes
    
    kv_savings = original_kv_memory - compressed_kv_memory
    kv_savings_ratio = kv_savings / original_kv_memory if original_kv_memory > 0 else 0
    
    # Calculate attention computation savings
    original_attn_ops = original_tokens ** 2
    compressed_attn_ops = compressed_tokens ** 2
    
    attn_savings = original_attn_ops - compressed_attn_ops
    attn_savings_ratio = attn_savings / original_attn_ops if original_attn_ops > 0 else 0
    
    return {
        "kv_memory_original_mb": original_kv_memory / (1024 * 1024),
        "kv_memory_compressed_mb": compressed_kv_memory / (1024 * 1024),
        "kv_memory_saved_mb": kv_savings / (1024 * 1024),
        "kv_savings_ratio": kv_savings_ratio,
        "attention_ops_original": original_attn_ops,
        "attention_ops_compressed": compressed_attn_ops,
        "attention_ops_saved": attn_savings,
        "attention_savings_ratio": attn_savings_ratio
    }


def create_visualization_data(clusters: List, assignments: Dict, 
                             compression_result: Dict) -> Dict:
    """Create data for visualizing SemToken results."""
    
    # Prepare cluster visualization data
    cluster_data = []
    for i, cluster in enumerate(clusters):
        cluster_info = {
            "id": i,
            "tokens": getattr(cluster, 'tokens', []),
            "size": getattr(cluster, 'size', 0),
            "coherence": getattr(cluster, 'coherence_score', 0),
            "span_start": getattr(cluster, 'span_start', 0),
            "span_end": getattr(cluster, 'span_end', 0)
        }
        cluster_data.append(cluster_info)
    
    # Prepare granularity visualization data
    granularity_data = []
    for cluster_id, assignment in assignments.items():
        granularity_info = {
            "cluster_id": cluster_id,
            "granularity": assignment.granularity.value,
            "entropy_score": assignment.entropy_score,
            "density_score": assignment.semantic_density,
            "importance_score": assignment.importance_score
        }
        granularity_data.append(granularity_info)
    
    # Prepare compression visualization data
    compression_data = {
        "original_length": compression_result.get("original_length", 0),
        "compressed_length": compression_result.get("compressed_length", 0),
        "compression_ratio": compression_result.get("compressed_length", 0) / 
                           max(1, compression_result.get("original_length", 1)),
        "spans_merged": compression_result.get("compression_stats", {}).get("spans_merged", 0),
        "tokens_saved": compression_result.get("compression_stats", {}).get("tokens_saved", 0)
    }
    
    return {
        "clusters": cluster_data,
        "granularity": granularity_data,
        "compression": compression_data,
        "timestamp": time.time()
    }


def save_results(results: Dict, filepath: str, format: str = "json"):
    """Save SemToken results to file."""
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    elif format == "pickle":
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Results saved to {filepath}")


def load_results(filepath: str, format: str = "json") -> Dict:
    """Load SemToken results from file."""
    
    if format == "json":
        with open(filepath, 'r') as f:
            return json.load(f)
    elif format == "pickle":
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported format: {format}")


def hash_text(text: str) -> str:
    """Create hash for text caching."""
    return hashlib.md5(text.encode()).hexdigest()


class ConfigManager:
    """Manages SemToken configuration and presets."""
    
    def __init__(self):
        self.presets = {
            "aggressive": {
                "compression_ratio": 0.3,
                "similarity_threshold": 0.8,
                "entropy_threshold": 0.6
            },
            "balanced": {
                "compression_ratio": 0.5,
                "similarity_threshold": 0.7,
                "entropy_threshold": 0.5
            },
            "conservative": {
                "compression_ratio": 0.7,
                "similarity_threshold": 0.6,
                "entropy_threshold": 0.4
            }
        }
    
    def get_preset(self, preset_name: str) -> Dict:
        """Get configuration preset."""
        return self.presets.get(preset_name, self.presets["balanced"])
    
    def save_config(self, config: Dict, filepath: str):
        """Save configuration to file."""
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_config(self, filepath: str) -> Dict:
        """Load configuration from file."""
        with open(filepath, 'r') as f:
            return json.load(f)
