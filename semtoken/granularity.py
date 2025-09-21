"""
Granularity assignment module for adaptive token granularity based on semantic density.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

from .clustering import ClusterInfo

logger = logging.getLogger(__name__)


class GranularityLevel(Enum):
    """Token granularity levels."""
    FINE = "fine"        # High-resolution tokenization for information-rich regions
    MEDIUM = "medium"    # Standard tokenization
    COARSE = "coarse"    # Low-resolution tokenization for repetitive/low-entropy regions


@dataclass
class GranularityInfo:
    """Information about granularity assignment for a token span."""
    cluster_id: int
    granularity: GranularityLevel
    semantic_density: float
    entropy_score: float
    importance_score: float
    reasoning: str  # Human-readable explanation for the assignment


class GranularityAssigner:
    """
    Assigns variable token granularity based on semantic density.
    
    Implements the entropy-based granularity assignment from the paper:
    - Fine-grained tokens for high-entropy, information-rich spans
    - Coarse-grained tokens for low-entropy, repetitive spans
    - Adaptive thresholds based on content analysis
    """
    
    def __init__(self, entropy_threshold: float = 0.5, density_window: int = 10,
                 fine_ratio: float = 0.3, coarse_ratio: float = 0.4,
                 adaptive_thresholding: bool = True):
        """
        Initialize granularity assigner.
        
        Args:
            entropy_threshold: Threshold for high/low entropy classification (δ in paper)
            density_window: Window size for computing local semantic density
            fine_ratio: Target ratio of fine-grained tokens
            coarse_ratio: Target ratio of coarse-grained tokens
            adaptive_thresholding: Whether to adapt thresholds based on input
        """
        self.entropy_threshold = entropy_threshold
        self.density_window = density_window
        self.fine_ratio = fine_ratio
        self.coarse_ratio = coarse_ratio
        self.adaptive_thresholding = adaptive_thresholding
        
        # Statistics for adaptive thresholding
        self.global_entropy_stats = {"mean": 0.5, "std": 0.2}
        
        logger.info(f"GranularityAssigner initialized with entropy_threshold={entropy_threshold}")
    
    def assign_granularity(self, clusters: List[ClusterInfo], 
                          embeddings: torch.Tensor) -> Dict[int, GranularityInfo]:
        """
        Assign granularity levels to clusters based on semantic density.
        
        Args:
            clusters: List of semantic clusters
            embeddings: Original token embeddings
            
        Returns:
            Dictionary mapping cluster IDs to granularity assignments
        """
        if not clusters:
            return {}
        
        # Compute semantic density and entropy for each cluster
        cluster_metrics = self._compute_cluster_metrics(clusters, embeddings)
        
        # Adapt thresholds if enabled
        if self.adaptive_thresholding:
            self._adapt_thresholds(cluster_metrics)
        
        # Assign granularity based on metrics
        granularity_assignments = {}
        
        for i, cluster in enumerate(clusters):
            metrics = cluster_metrics[i]
            granularity_info = self._assign_cluster_granularity(i, cluster, metrics)
            granularity_assignments[i] = granularity_info
        
        # Post-process to ensure target ratios
        granularity_assignments = self._balance_granularity_ratios(
            granularity_assignments, clusters
        )
        
        logger.debug(f"Assigned granularity to {len(clusters)} clusters")
        return granularity_assignments
    
    def _compute_cluster_metrics(self, clusters: List[ClusterInfo], 
                                embeddings: torch.Tensor) -> List[Dict]:
        """Compute semantic metrics for each cluster."""
        cluster_metrics = []
        
        for cluster in clusters:
            # Get embeddings for this cluster
            cluster_embeddings = embeddings[cluster.indices]
            
            # Compute semantic entropy (trace of covariance matrix)
            entropy_score = self._compute_semantic_entropy(cluster_embeddings)
            
            # Compute local semantic density
            density_score = self._compute_semantic_density(
                cluster, embeddings, self.density_window
            )
            
            # Compute importance score (combination of entropy and density)
            importance_score = self._compute_importance_score(
                entropy_score, density_score, cluster
            )
            
            # Additional metrics
            cluster_coherence = cluster.coherence_score
            cluster_size = cluster.size
            span_length = cluster.span_end - cluster.span_start + 1
            
            metrics = {
                "entropy_score": entropy_score,
                "density_score": density_score,
                "importance_score": importance_score,
                "coherence_score": cluster_coherence,
                "cluster_size": cluster_size,
                "span_length": span_length,
                "tokens": cluster.tokens
            }
            
            cluster_metrics.append(metrics)
        
        return cluster_metrics
    
    def _compute_semantic_entropy(self, embeddings: torch.Tensor) -> float:
        """
        Compute semantic entropy using trace of covariance matrix.
        
        H(T) = Tr(Cov({f_θ(x_i) | x_i ∈ T}))
        """
        if embeddings.size(0) <= 1:
            return 1.0  # Maximum entropy for single embedding
        
        # Center the embeddings
        centered = embeddings - embeddings.mean(dim=0, keepdim=True)
        
        # Compute covariance matrix
        n = embeddings.size(0)
        cov_matrix = torch.mm(centered.t(), centered) / (n - 1)
        
        # Return normalized trace
        entropy = torch.trace(cov_matrix).item()
        
        # Normalize by embedding dimension for comparable scores
        return entropy / embeddings.size(1)
    
    def _compute_semantic_density(self, cluster: ClusterInfo, embeddings: torch.Tensor, 
                                 window_size: int) -> float:
        """
        Compute local semantic density around the cluster.
        
        Measures how much semantic information is concentrated in the local region.
        """
        # Define local window around cluster
        center_pos = (cluster.span_start + cluster.span_end) // 2
        window_start = max(0, center_pos - window_size // 2)
        window_end = min(embeddings.size(0), center_pos + window_size // 2)
        
        # Get embeddings in the window
        window_embeddings = embeddings[window_start:window_end]
        
        if window_embeddings.size(0) <= 1:
            return 0.5  # Default density for edge cases
        
        # Compute density as variance of embeddings in the window
        density = self._compute_semantic_entropy(window_embeddings)
        
        # Boost density for clusters with unique content
        uniqueness_boost = self._compute_uniqueness_score(cluster, embeddings)
        
        return min(1.0, density * (1.0 + uniqueness_boost))
    
    def _compute_uniqueness_score(self, cluster: ClusterInfo, 
                                 embeddings: torch.Tensor) -> float:
        """
        Compute how unique this cluster is compared to the rest of the sequence.
        """
        cluster_centroid = cluster.centroid
        
        # Compute similarity to all other embeddings
        other_indices = [i for i in range(embeddings.size(0)) if i not in cluster.indices]
        
        if not other_indices:
            return 1.0  # Maximally unique if no other tokens
        
        other_embeddings = embeddings[other_indices]
        
        # Compute average similarity to other embeddings
        similarities = torch.cosine_similarity(
            cluster_centroid.unsqueeze(0), other_embeddings, dim=1
        )
        
        avg_similarity = similarities.mean().item()
        
        # Uniqueness is inverse of similarity
        return 1.0 - avg_similarity
    
    def _compute_importance_score(self, entropy_score: float, density_score: float, 
                                 cluster: ClusterInfo) -> float:
        """
        Compute overall importance score for granularity assignment.
        
        Combines entropy, density, and other factors into a single score.
        """
        # Base score from entropy and density
        base_score = 0.6 * entropy_score + 0.4 * density_score
        
        # Boost for longer clusters (more context)
        length_boost = min(0.2, cluster.size * 0.05)
        
        # Boost for high coherence clusters
        coherence_boost = max(0, (cluster.coherence_score - 0.7) * 0.3)
        
        # Penalty for very small clusters
        size_penalty = max(0, (3 - cluster.size) * 0.1)
        
        importance = base_score + length_boost + coherence_boost - size_penalty
        
        return max(0.0, min(1.0, importance))
    
    def _assign_cluster_granularity(self, cluster_id: int, cluster: ClusterInfo, 
                                   metrics: Dict) -> GranularityInfo:
        """Assign granularity level to a single cluster."""
        
        entropy_score = metrics["entropy_score"]
        density_score = metrics["density_score"]
        importance_score = metrics["importance_score"]
        
        # Determine granularity based on thresholds
        if entropy_score > self.entropy_threshold:
            if importance_score > 0.7:
                granularity = GranularityLevel.FINE
                reasoning = f"High entropy ({entropy_score:.3f}) and importance ({importance_score:.3f})"
            else:
                granularity = GranularityLevel.MEDIUM
                reasoning = f"High entropy ({entropy_score:.3f}) but moderate importance"
        else:
            if density_score < 0.3:
                granularity = GranularityLevel.COARSE
                reasoning = f"Low entropy ({entropy_score:.3f}) and density ({density_score:.3f})"
            else:
                granularity = GranularityLevel.MEDIUM
                reasoning = f"Low entropy but moderate density ({density_score:.3f})"
        
        # Special cases
        if self._is_special_content(cluster.tokens):
            granularity = GranularityLevel.FINE
            reasoning = "Special content (numbers, names, technical terms)"
        
        return GranularityInfo(
            cluster_id=cluster_id,
            granularity=granularity,
            semantic_density=density_score,
            entropy_score=entropy_score,
            importance_score=importance_score,
            reasoning=reasoning
        )
    
    def _is_special_content(self, tokens: List[str]) -> bool:
        """
        Identify special content that should always use fine granularity.
        
        Examples: numbers, proper names, technical terms, code snippets
        """
        special_patterns = [
            lambda t: t.isdigit(),  # Numbers
            lambda t: t[0].isupper() and len(t) > 1,  # Proper names
            lambda t: any(c in t for c in "()[]{}"),  # Code-like content
            lambda t: t.startswith("#") or t.startswith("@"),  # Special markers
            lambda t: len(t) > 10 and not t.isalpha()  # Complex tokens
        ]
        
        for token in tokens:
            if any(pattern(token) for pattern in special_patterns):
                return True
        
        return False
    
    def _adapt_thresholds(self, cluster_metrics: List[Dict]):
        """Adapt thresholds based on input characteristics."""
        
        if not cluster_metrics:
            return
        
        # Compute global statistics
        entropy_scores = [m["entropy_score"] for m in cluster_metrics]
        density_scores = [m["density_score"] for m in cluster_metrics]
        
        entropy_mean = np.mean(entropy_scores)
        entropy_std = np.std(entropy_scores)
        
        # Update global statistics with exponential moving average
        alpha = 0.1
        self.global_entropy_stats["mean"] = (
            alpha * entropy_mean + (1 - alpha) * self.global_entropy_stats["mean"]
        )
        self.global_entropy_stats["std"] = (
            alpha * entropy_std + (1 - alpha) * self.global_entropy_stats["std"]
        )
        
        # Adapt entropy threshold based on distribution
        if entropy_std > 0.1:  # High variance - use relative threshold
            self.entropy_threshold = entropy_mean + 0.5 * entropy_std
        else:  # Low variance - use absolute threshold
            self.entropy_threshold = max(0.4, min(0.7, entropy_mean + 0.1))
        
        logger.debug(f"Adapted entropy threshold to {self.entropy_threshold:.3f}")
    
    def _balance_granularity_ratios(self, assignments: Dict[int, GranularityInfo],
                                   clusters: List[ClusterInfo]) -> Dict[int, GranularityInfo]:
        """
        Balance granularity assignments to meet target ratios.
        
        Ensures we don't have too many fine or coarse-grained tokens.
        """
        if not assignments:
            return assignments
        
        # Count current assignments
        granularity_counts = {level: 0 for level in GranularityLevel}
        total_tokens = 0
        
        for assignment in assignments.values():
            granularity_counts[assignment.granularity] += 1
            total_tokens += 1
        
        # Compute current ratios
        current_fine_ratio = granularity_counts[GranularityLevel.FINE] / total_tokens
        current_coarse_ratio = granularity_counts[GranularityLevel.COARSE] / total_tokens
        
        # Adjust if ratios are too extreme
        if current_fine_ratio > self.fine_ratio * 1.5:
            # Too many fine-grained - demote some to medium
            assignments = self._demote_excess_fine(assignments, clusters)
        
        if current_coarse_ratio > self.coarse_ratio * 1.5:
            # Too many coarse-grained - promote some to medium
            assignments = self._promote_excess_coarse(assignments, clusters)
        
        return assignments
    
    def _demote_excess_fine(self, assignments: Dict[int, GranularityInfo],
                           clusters: List[ClusterInfo]) -> Dict[int, GranularityInfo]:
        """Demote some fine-grained assignments to medium."""
        
        # Find fine-grained assignments sorted by importance (ascending)
        fine_assignments = [
            (cid, info) for cid, info in assignments.items() 
            if info.granularity == GranularityLevel.FINE
        ]
        fine_assignments.sort(key=lambda x: x[1].importance_score)
        
        # Demote lowest importance fine-grained assignments
        num_to_demote = len(fine_assignments) // 4  # Demote 25%
        
        for i in range(num_to_demote):
            cid, info = fine_assignments[i]
            assignments[cid] = GranularityInfo(
                cluster_id=info.cluster_id,
                granularity=GranularityLevel.MEDIUM,
                semantic_density=info.semantic_density,
                entropy_score=info.entropy_score,
                importance_score=info.importance_score,
                reasoning=f"{info.reasoning} (demoted for ratio balancing)"
            )
        
        return assignments
    
    def _promote_excess_coarse(self, assignments: Dict[int, GranularityInfo],
                              clusters: List[ClusterInfo]) -> Dict[int, GranularityInfo]:
        """Promote some coarse-grained assignments to medium."""
        
        # Find coarse-grained assignments sorted by importance (descending)
        coarse_assignments = [
            (cid, info) for cid, info in assignments.items() 
            if info.granularity == GranularityLevel.COARSE
        ]
        coarse_assignments.sort(key=lambda x: x[1].importance_score, reverse=True)
        
        # Promote highest importance coarse-grained assignments
        num_to_promote = len(coarse_assignments) // 4  # Promote 25%
        
        for i in range(num_to_promote):
            cid, info = coarse_assignments[i]
            assignments[cid] = GranularityInfo(
                cluster_id=info.cluster_id,
                granularity=GranularityLevel.MEDIUM,
                semantic_density=info.semantic_density,
                entropy_score=info.entropy_score,
                importance_score=info.importance_score,
                reasoning=f"{info.reasoning} (promoted for ratio balancing)"
            )
        
        return assignments
    
    def get_granularity_stats(self, assignments: Dict[int, GranularityInfo]) -> Dict:
        """Get statistics about granularity assignments."""
        
        if not assignments:
            return {"total": 0, "ratios": {}, "avg_scores": {}}
        
        # Count assignments by granularity
        granularity_counts = {level.value: 0 for level in GranularityLevel}
        entropy_scores = []
        density_scores = []
        importance_scores = []
        
        for assignment in assignments.values():
            granularity_counts[assignment.granularity.value] += 1
            entropy_scores.append(assignment.entropy_score)
            density_scores.append(assignment.semantic_density)
            importance_scores.append(assignment.importance_score)
        
        total = len(assignments)
        ratios = {level: count / total for level, count in granularity_counts.items()}
        
        return {
            "total": total,
            "counts": granularity_counts,
            "ratios": ratios,
            "avg_scores": {
                "entropy": np.mean(entropy_scores),
                "density": np.mean(density_scores),
                "importance": np.mean(importance_scores)
            },
            "score_std": {
                "entropy": np.std(entropy_scores),
                "density": np.std(density_scores),
                "importance": np.std(importance_scores)
            }
        }


class DynamicGranularityAssigner(GranularityAssigner):
    """
    Dynamic granularity assigner that adapts to content type and context.
    
    Extends base functionality with:
    - Content-type detection (narrative, technical, conversational)
    - Context-aware thresholding
    - Progressive refinement based on feedback
    """
    
    def __init__(self, *args, content_type: str = "auto", **kwargs):
        super().__init__(*args, **kwargs)
        self.content_type = content_type
        self.refinement_history = []
    
    def assign_granularity(self, clusters: List[ClusterInfo], 
                          embeddings: torch.Tensor) -> Dict[int, GranularityInfo]:
        """Dynamic granularity assignment with content-type adaptation."""
        
        # Detect content type if auto
        if self.content_type == "auto":
            detected_type = self._detect_content_type(clusters, embeddings)
        else:
            detected_type = self.content_type
        
        # Adapt parameters for content type
        self._adapt_for_content_type(detected_type)
        
        # Perform base assignment
        assignments = super().assign_granularity(clusters, embeddings)
        
        # Apply content-specific post-processing
        assignments = self._post_process_for_content_type(assignments, detected_type)
        
        return assignments
    
    def _detect_content_type(self, clusters: List[ClusterInfo], 
                            embeddings: torch.Tensor) -> str:
        """Detect the type of content for adaptive processing."""
        
        # Analyze token characteristics
        all_tokens = []
        for cluster in clusters:
            all_tokens.extend(cluster.tokens)
        
        # Compute content features
        avg_token_length = np.mean([len(token) for token in all_tokens])
        num_punct = sum(1 for token in all_tokens if token in ".,!?;:")
        num_numbers = sum(1 for token in all_tokens if token.isdigit())
        num_technical = sum(1 for token in all_tokens if any(c in token for c in "()[]{}"))
        
        punct_ratio = num_punct / len(all_tokens)
        number_ratio = num_numbers / len(all_tokens)
        technical_ratio = num_technical / len(all_tokens)
        
        # Classification rules
        if technical_ratio > 0.1 or avg_token_length > 6:
            return "technical"
        elif punct_ratio > 0.05 and avg_token_length < 5:
            return "conversational"
        else:
            return "narrative"
    
    def _adapt_for_content_type(self, content_type: str):
        """Adapt parameters based on detected content type."""
        
        if content_type == "technical":
            # Technical content - more fine-grained for precision
            self.fine_ratio = 0.4
            self.coarse_ratio = 0.3
            self.entropy_threshold *= 0.9
            
        elif content_type == "conversational":
            # Conversational - more coarse-grained for efficiency
            self.fine_ratio = 0.2
            self.coarse_ratio = 0.5
            self.entropy_threshold *= 1.1
            
        elif content_type == "narrative":
            # Narrative - balanced approach
            self.fine_ratio = 0.3
            self.coarse_ratio = 0.4
            # Keep default entropy threshold
        
        logger.debug(f"Adapted for content type: {content_type}")
    
    def _post_process_for_content_type(self, assignments: Dict[int, GranularityInfo],
                                      content_type: str) -> Dict[int, GranularityInfo]:
        """Apply content-type specific post-processing."""
        
        if content_type == "technical":
            # Ensure technical terms get fine granularity
            for cid, assignment in assignments.items():
                if self._contains_technical_terms(assignment):
                    assignments[cid] = GranularityInfo(
                        cluster_id=assignment.cluster_id,
                        granularity=GranularityLevel.FINE,
                        semantic_density=assignment.semantic_density,
                        entropy_score=assignment.entropy_score,
                        importance_score=assignment.importance_score,
                        reasoning=f"{assignment.reasoning} (technical content override)"
                    )
        
        return assignments
    
    def _contains_technical_terms(self, assignment: GranularityInfo) -> bool:
        """Check if assignment contains technical terminology."""
        # This would typically use a more sophisticated technical term detector
        # For now, use simple heuristics
        return assignment.entropy_score > 0.7 and assignment.importance_score > 0.6


# Utility functions for granularity visualization and analysis
def visualize_granularity_distribution(assignments: Dict[int, GranularityInfo]) -> Dict:
    """
    Create visualization data for granularity distribution.
    
    Returns data suitable for plotting granularity patterns.
    """
    if not assignments:
        return {"positions": [], "granularities": [], "scores": []}
    
    positions = []
    granularities = []
    entropy_scores = []
    density_scores = []
    
    for assignment in assignments.values():
        positions.append(assignment.cluster_id)
        granularities.append(assignment.granularity.value)
        entropy_scores.append(assignment.entropy_score)
        density_scores.append(assignment.semantic_density)
    
    return {
        "positions": positions,
        "granularities": granularities,
        "entropy_scores": entropy_scores,
        "density_scores": density_scores,
        "reasoning": [a.reasoning for a in assignments.values()]
    }
