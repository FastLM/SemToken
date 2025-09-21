"""
Budget allocation module for intelligent token selection under compression constraints.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import heapq
import logging

from .clustering import ClusterInfo
from .granularity import GranularityInfo, GranularityLevel

logger = logging.getLogger(__name__)


@dataclass
class BudgetAllocation:
    """Result of budget allocation process."""
    selected_clusters: List[int]  # IDs of selected clusters
    allocation_scores: Dict[int, float]  # Score for each cluster
    total_budget_used: int  # Total tokens allocated
    compression_achieved: float  # Actual compression ratio
    selection_reasoning: Dict[int, str]  # Reasoning for each selection


class BudgetAllocator:
    """
    Budget-aware token allocation system.
    
    Implements the entropy-guided selection under budget constraint from the paper:
    max_{X' ⊆ X, |X'| ≤ B} Σ_{x'_i ∈ X'} H(x'_i)
    
    Selects highest-entropy segments to retain within the given budget.
    """
    
    def __init__(self, compression_ratio: float = 0.5, max_budget: Optional[int] = None,
                 priority_boost: float = 1.2, query_weight: float = 0.3,
                 diversity_penalty: float = 0.1):
        """
        Initialize budget allocator.
        
        Args:
            compression_ratio: Target compression ratio (0.5 = 50% compression)
            max_budget: Maximum absolute token budget (overrides ratio if set)
            priority_boost: Boost factor for high-priority content
            query_weight: Weight for query-aware scoring
            diversity_penalty: Penalty for selecting too similar content
        """
        self.compression_ratio = compression_ratio
        self.max_budget = max_budget
        self.priority_boost = priority_boost
        self.query_weight = query_weight
        self.diversity_penalty = diversity_penalty
        
        # Statistics for adaptive allocation
        self.allocation_history = []
        
        logger.info(f"BudgetAllocator initialized with compression_ratio={compression_ratio}")
    
    def allocate_budget(self, clusters: List[ClusterInfo], 
                       granularity_assignments: Dict[int, GranularityInfo],
                       query_embedding: Optional[torch.Tensor] = None) -> BudgetAllocation:
        """
        Allocate budget across clusters to maximize information retention.
        
        Args:
            clusters: List of semantic clusters
            granularity_assignments: Granularity information for each cluster
            query_embedding: Optional query embedding for query-aware allocation
            
        Returns:
            BudgetAllocation with selected clusters and allocation details
        """
        if not clusters:
            return BudgetAllocation([], {}, 0, 0.0, {})
        
        # Calculate budget
        total_tokens = sum(cluster.size for cluster in clusters)
        if self.max_budget:
            budget = min(self.max_budget, total_tokens)
        else:
            budget = int(total_tokens * self.compression_ratio)
        
        logger.debug(f"Allocating budget: {budget} tokens from {total_tokens} total")
        
        # Compute allocation scores for each cluster
        allocation_scores = self._compute_allocation_scores(
            clusters, granularity_assignments, query_embedding
        )
        
        # Select clusters using budget-aware algorithm
        selected_clusters, selection_reasoning = self._select_clusters(
            clusters, allocation_scores, budget
        )
        
        # Calculate final statistics
        total_budget_used = sum(clusters[cid].size for cid in selected_clusters)
        compression_achieved = total_budget_used / total_tokens
        
        allocation = BudgetAllocation(
            selected_clusters=selected_clusters,
            allocation_scores=allocation_scores,
            total_budget_used=total_budget_used,
            compression_achieved=compression_achieved,
            selection_reasoning=selection_reasoning
        )
        
        # Update history for adaptive learning
        self.allocation_history.append({
            "budget": budget,
            "selected": len(selected_clusters),
            "compression": compression_achieved
        })
        
        logger.info(f"Budget allocation: {len(selected_clusters)} clusters selected, "
                   f"compression: {compression_achieved:.2%}")
        
        return allocation
    
    def _compute_allocation_scores(self, clusters: List[ClusterInfo],
                                  granularity_assignments: Dict[int, GranularityInfo],
                                  query_embedding: Optional[torch.Tensor]) -> Dict[int, float]:
        """Compute allocation scores for each cluster."""
        
        scores = {}
        
        for i, cluster in enumerate(clusters):
            # Base score from semantic entropy
            base_score = self._compute_entropy_score(cluster)
            
            # Granularity boost
            granularity_boost = self._compute_granularity_boost(
                granularity_assignments.get(i)
            )
            
            # Query relevance boost
            query_boost = self._compute_query_boost(cluster, query_embedding)
            
            # Position boost (favor diverse positions)
            position_boost = self._compute_position_boost(cluster, clusters)
            
            # Coherence boost
            coherence_boost = self._compute_coherence_boost(cluster)
            
            # Size efficiency (tokens per unit of information)
            size_efficiency = self._compute_size_efficiency(cluster)
            
            # Combine all factors
            total_score = (
                base_score * 
                (1.0 + granularity_boost) * 
                (1.0 + query_boost) * 
                (1.0 + position_boost) * 
                (1.0 + coherence_boost) * 
                size_efficiency
            )
            
            scores[i] = total_score
        
        return scores
    
    def _compute_entropy_score(self, cluster: ClusterInfo) -> float:
        """Compute base entropy score for cluster."""
        
        # Use cluster coherence as proxy for entropy
        # Higher coherence = more structured = lower entropy for redundancy
        # But we want to keep high-information content
        
        # Invert coherence for entropy-like measure
        entropy_proxy = 1.0 - cluster.coherence_score
        
        # Boost for larger clusters (more context)
        size_factor = min(2.0, 1.0 + cluster.size * 0.1)
        
        return entropy_proxy * size_factor
    
    def _compute_granularity_boost(self, granularity_info: Optional[GranularityInfo]) -> float:
        """Compute boost based on granularity assignment."""
        
        if not granularity_info:
            return 0.0
        
        # Boost fine-grained content
        if granularity_info.granularity == GranularityLevel.FINE:
            return self.priority_boost - 1.0
        elif granularity_info.granularity == GranularityLevel.MEDIUM:
            return (self.priority_boost - 1.0) * 0.5
        else:  # COARSE
            return 0.0
    
    def _compute_query_boost(self, cluster: ClusterInfo, 
                            query_embedding: Optional[torch.Tensor]) -> float:
        """Compute boost based on query relevance."""
        
        if query_embedding is None:
            return 0.0
        
        # Compute similarity between cluster centroid and query
        similarity = torch.cosine_similarity(
            cluster.centroid.unsqueeze(0), 
            query_embedding.unsqueeze(0)
        ).item()
        
        # Convert similarity to boost (0 to query_weight)
        return max(0.0, similarity * self.query_weight)
    
    def _compute_position_boost(self, cluster: ClusterInfo, 
                               all_clusters: List[ClusterInfo]) -> float:
        """Compute boost for positional diversity."""
        
        # Favor clusters that are spread out across the sequence
        total_span = max(c.span_end for c in all_clusters) - min(c.span_start for c in all_clusters)
        
        if total_span == 0:
            return 0.0
        
        # Boost clusters at beginning and end
        relative_start = cluster.span_start / total_span
        relative_end = cluster.span_end / total_span
        
        # U-shaped boost (higher at extremes)
        edge_boost = max(1.0 - relative_start, relative_end) * 0.1
        
        return edge_boost
    
    def _compute_coherence_boost(self, cluster: ClusterInfo) -> float:
        """Compute boost based on cluster coherence."""
        
        # Moderate boost for high coherence (well-formed clusters)
        if cluster.coherence_score > 0.8:
            return 0.15
        elif cluster.coherence_score > 0.6:
            return 0.1
        else:
            return 0.0
    
    def _compute_size_efficiency(self, cluster: ClusterInfo) -> float:
        """Compute size efficiency score."""
        
        # Favor clusters with good information density
        # Penalize very large clusters (might be too coarse)
        # Penalize very small clusters (might not be worth keeping)
        
        if cluster.size <= 1:
            return 0.5  # Single tokens are less efficient
        elif cluster.size <= 3:
            return 1.0  # Optimal size
        elif cluster.size <= 6:
            return 0.9  # Still good
        else:
            return 0.7  # Larger clusters are less efficient
    
    def _select_clusters(self, clusters: List[ClusterInfo], 
                        allocation_scores: Dict[int, float],
                        budget: int) -> Tuple[List[int], Dict[int, str]]:
        """
        Select clusters using budget constraint.
        
        Uses a greedy algorithm with diversity considerations.
        """
        # Create priority queue with (negative_score, cluster_id, cluster_size)
        # Use negative score for max-heap behavior
        priority_queue = []
        
        for cluster_id, score in allocation_scores.items():
            cluster = clusters[cluster_id]
            heapq.heappush(priority_queue, (-score, cluster_id, cluster.size))
        
        selected_clusters = []
        selection_reasoning = {}
        remaining_budget = budget
        selected_positions = set()
        
        while priority_queue and remaining_budget > 0:
            neg_score, cluster_id, cluster_size = heapq.heappop(priority_queue)
            score = -neg_score
            cluster = clusters[cluster_id]
            
            # Check if we can afford this cluster
            if cluster_size > remaining_budget:
                selection_reasoning[cluster_id] = f"Skipped: exceeds remaining budget ({cluster_size} > {remaining_budget})"
                continue
            
            # Check diversity constraint
            diversity_penalty = self._compute_diversity_penalty(
                cluster, selected_positions
            )
            
            adjusted_score = score * (1.0 - diversity_penalty)
            
            # Apply threshold for selection
            if adjusted_score < 0.1:  # Minimum score threshold
                selection_reasoning[cluster_id] = f"Skipped: score too low ({adjusted_score:.3f})"
                continue
            
            # Select this cluster
            selected_clusters.append(cluster_id)
            remaining_budget -= cluster_size
            selected_positions.update(range(cluster.span_start, cluster.span_end + 1))
            
            selection_reasoning[cluster_id] = (
                f"Selected: score={score:.3f}, adjusted={adjusted_score:.3f}, "
                f"size={cluster_size}, budget_remaining={remaining_budget}"
            )
            
            logger.debug(f"Selected cluster {cluster_id}: {selection_reasoning[cluster_id]}")
        
        return selected_clusters, selection_reasoning
    
    def _compute_diversity_penalty(self, cluster: ClusterInfo, 
                                  selected_positions: Set[int]) -> float:
        """Compute penalty for selecting overlapping content."""
        
        cluster_positions = set(range(cluster.span_start, cluster.span_end + 1))
        overlap = len(cluster_positions.intersection(selected_positions))
        
        if overlap == 0:
            return 0.0  # No penalty for non-overlapping content
        
        # Penalty proportional to overlap
        overlap_ratio = overlap / len(cluster_positions)
        return overlap_ratio * self.diversity_penalty
    
    def adaptive_budget_adjustment(self, performance_feedback: Dict) -> None:
        """
        Adaptively adjust budget allocation based on performance feedback.
        
        Args:
            performance_feedback: Dictionary with performance metrics
        """
        if not self.allocation_history:
            return
        
        # Extract feedback metrics
        quality_score = performance_feedback.get("quality_score", 0.5)
        efficiency_score = performance_feedback.get("efficiency_score", 0.5)
        user_satisfaction = performance_feedback.get("user_satisfaction", 0.5)
        
        # Compute overall performance
        overall_performance = (quality_score + efficiency_score + user_satisfaction) / 3
        
        # Adjust compression ratio based on performance
        if overall_performance > 0.8:
            # Good performance - can compress more aggressively
            self.compression_ratio = max(0.3, self.compression_ratio - 0.05)
        elif overall_performance < 0.5:
            # Poor performance - be more conservative
            self.compression_ratio = min(0.8, self.compression_ratio + 0.05)
        
        logger.info(f"Adjusted compression ratio to {self.compression_ratio:.2f} "
                   f"based on performance: {overall_performance:.2f}")
    
    def get_budget_stats(self) -> Dict:
        """Get statistics about budget allocation history."""
        
        if not self.allocation_history:
            return {"total_allocations": 0}
        
        compressions = [h["compression"] for h in self.allocation_history]
        budgets = [h["budget"] for h in self.allocation_history]
        selections = [h["selected"] for h in self.allocation_history]
        
        return {
            "total_allocations": len(self.allocation_history),
            "avg_compression": np.mean(compressions),
            "avg_budget": np.mean(budgets),
            "avg_selections": np.mean(selections),
            "compression_std": np.std(compressions),
            "current_compression_ratio": self.compression_ratio
        }


class QueryAwareBudgetAllocator(BudgetAllocator):
    """
    Query-aware budget allocator with autoregressive merging support.
    
    Implements query conditioning from the paper:
    s_i = sim(q_t, h̄_i), where h̄_i = mean({h_j ∈ x'_i})
    """
    
    def __init__(self, *args, query_decay: float = 0.9, 
                 context_window: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.query_decay = query_decay
        self.context_window = context_window
        
        # History of query-cluster interactions
        self.query_history = []
    
    def allocate_budget_autoregressive(self, clusters: List[ClusterInfo],
                                     granularity_assignments: Dict[int, GranularityInfo],
                                     query_sequence: List[torch.Tensor],
                                     generation_step: int) -> BudgetAllocation:
        """
        Allocate budget for autoregressive generation with query conditioning.
        
        Args:
            clusters: List of semantic clusters
            granularity_assignments: Granularity assignments
            query_sequence: Sequence of query embeddings over time
            generation_step: Current generation step
            
        Returns:
            BudgetAllocation optimized for current generation context
        """
        # Get current and historical query embeddings
        current_query = query_sequence[generation_step] if generation_step < len(query_sequence) else None
        
        # Compute time-weighted query embedding
        weighted_query = self._compute_weighted_query(query_sequence, generation_step)
        
        # Compute backward importance scores
        backward_importance = self._compute_backward_importance(
            clusters, weighted_query
        )
        
        # Adjust allocation scores with temporal weighting
        base_allocation = super().allocate_budget(
            clusters, granularity_assignments, weighted_query
        )
        
        # Apply temporal adjustments
        adjusted_scores = {}
        for cluster_id, base_score in base_allocation.allocation_scores.items():
            temporal_weight = self._compute_temporal_weight(
                clusters[cluster_id], generation_step
            )
            backward_score = backward_importance.get(cluster_id, 0.0)
            
            adjusted_score = base_score * temporal_weight + backward_score * 0.3
            adjusted_scores[cluster_id] = adjusted_score
        
        # Re-select with adjusted scores
        budget = int(sum(c.size for c in clusters) * self.compression_ratio)
        selected_clusters, reasoning = self._select_clusters(
            clusters, adjusted_scores, budget
        )
        
        return BudgetAllocation(
            selected_clusters=selected_clusters,
            allocation_scores=adjusted_scores,
            total_budget_used=sum(clusters[cid].size for cid in selected_clusters),
            compression_achieved=sum(clusters[cid].size for cid in selected_clusters) / sum(c.size for c in clusters),
            selection_reasoning=reasoning
        )
    
    def _compute_weighted_query(self, query_sequence: List[torch.Tensor], 
                               current_step: int) -> torch.Tensor:
        """Compute time-weighted query embedding."""
        
        if not query_sequence or current_step >= len(query_sequence):
            return torch.zeros(384)  # Default embedding size
        
        # Weight recent queries more heavily
        weighted_sum = torch.zeros_like(query_sequence[0])
        total_weight = 0.0
        
        for i in range(max(0, current_step - self.context_window), current_step + 1):
            if i < len(query_sequence):
                weight = self.query_decay ** (current_step - i)
                weighted_sum += weight * query_sequence[i]
                total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return query_sequence[current_step]
    
    def _compute_backward_importance(self, clusters: List[ClusterInfo],
                                   query_embedding: torch.Tensor) -> Dict[int, float]:
        """Compute backward importance of past token spans."""
        
        importance_scores = {}
        
        for i, cluster in enumerate(clusters):
            # Similarity between query and cluster centroid
            similarity = torch.cosine_similarity(
                query_embedding.unsqueeze(0),
                cluster.centroid.unsqueeze(0)
            ).item()
            
            # Apply threshold filtering
            if similarity > 0.5:  # Threshold for relevance
                importance_scores[i] = similarity
            else:
                importance_scores[i] = 0.0
        
        return importance_scores
    
    def _compute_temporal_weight(self, cluster: ClusterInfo, generation_step: int) -> float:
        """Compute temporal weight based on cluster position and generation step."""
        
        # Recent clusters get higher weight
        # This is a simplified version - in practice would depend on actual positions
        
        # For now, use cluster ID as proxy for temporal position
        temporal_distance = abs(generation_step - cluster.span_start)
        
        # Exponential decay with distance
        temporal_weight = np.exp(-temporal_distance * 0.1)
        
        return max(0.1, temporal_weight)  # Minimum weight


# Utility functions for budget optimization
def optimize_budget_allocation(clusters: List[ClusterInfo], 
                              target_compression: float,
                              quality_weights: Dict[str, float] = None) -> Dict:
    """
    Optimize budget allocation using more sophisticated algorithms.
    
    Args:
        clusters: List of clusters to allocate budget to
        target_compression: Target compression ratio
        quality_weights: Weights for different quality metrics
        
    Returns:
        Optimized allocation strategy
    """
    if quality_weights is None:
        quality_weights = {
            "entropy": 0.4,
            "coherence": 0.3,
            "size_efficiency": 0.2,
            "diversity": 0.1
        }
    
    # This would implement more sophisticated optimization
    # For now, return a simple greedy strategy
    
    total_tokens = sum(c.size for c in clusters)
    budget = int(total_tokens * target_compression)
    
    # Score each cluster
    cluster_scores = []
    for i, cluster in enumerate(clusters):
        entropy_score = 1.0 - cluster.coherence_score  # Proxy for entropy
        coherence_score = cluster.coherence_score
        size_efficiency = 1.0 / cluster.size if cluster.size > 0 else 0
        diversity_score = 1.0  # Would compute based on position diversity
        
        total_score = (
            quality_weights["entropy"] * entropy_score +
            quality_weights["coherence"] * coherence_score +
            quality_weights["size_efficiency"] * size_efficiency +
            quality_weights["diversity"] * diversity_score
        )
        
        cluster_scores.append((total_score, i, cluster.size))
    
    # Sort by score (descending)
    cluster_scores.sort(reverse=True)
    
    # Greedy selection
    selected = []
    remaining_budget = budget
    
    for score, cluster_id, size in cluster_scores:
        if size <= remaining_budget:
            selected.append(cluster_id)
            remaining_budget -= size
    
    return {
        "selected_clusters": selected,
        "budget_used": budget - remaining_budget,
        "compression_achieved": (budget - remaining_budget) / total_tokens,
        "optimization_method": "greedy"
    }
