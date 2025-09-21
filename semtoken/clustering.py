"""
Semantic clustering module for grouping semantically equivalent tokens.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


@dataclass
class ClusterInfo:
    """Information about a semantic cluster."""
    indices: List[int]  # Token indices in the cluster
    tokens: List[str]  # Actual token strings
    centroid: torch.Tensor  # Cluster centroid embedding
    coherence_score: float  # Internal coherence measure
    size: int  # Number of tokens in cluster
    span_start: int  # Start position in original sequence
    span_end: int  # End position in original sequence


class SemanticCluster:
    """
    Semantic clustering component for identifying and merging similar token spans.
    
    Implements the local similarity-based clustering from Algorithm 1:
    - Greedy span formation based on cosine similarity
    - Configurable similarity thresholds
    - Support for different clustering algorithms
    """
    
    def __init__(self, similarity_threshold: float = 0.7, min_cluster_size: int = 2,
                 max_cluster_size: int = 10, clustering_method: str = "greedy",
                 coherence_threshold: float = 0.6):
        """
        Initialize semantic clustering.
        
        Args:
            similarity_threshold: Minimum similarity for token grouping (τ in paper)
            min_cluster_size: Minimum tokens per cluster
            max_cluster_size: Maximum tokens per cluster
            clustering_method: Clustering algorithm ("greedy", "agglomerative", "dbscan")
            coherence_threshold: Minimum coherence score for valid clusters
        """
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.clustering_method = clustering_method
        self.coherence_threshold = coherence_threshold
        
        logger.info(f"SemanticCluster initialized with method={clustering_method}, "
                   f"threshold={similarity_threshold}")
    
    def cluster_tokens(self, tokens: List[str], embeddings: torch.Tensor) -> List[ClusterInfo]:
        """
        Main clustering function that groups semantically similar tokens.
        
        Args:
            tokens: List of token strings
            embeddings: Token embeddings tensor [num_tokens, embedding_dim]
            
        Returns:
            List of ClusterInfo objects representing semantic clusters
        """
        if len(tokens) != embeddings.size(0):
            raise ValueError("Number of tokens must match number of embeddings")
        
        if self.clustering_method == "greedy":
            return self._greedy_clustering(tokens, embeddings)
        elif self.clustering_method == "agglomerative":
            return self._agglomerative_clustering(tokens, embeddings)
        elif self.clustering_method == "dbscan":
            return self._dbscan_clustering(tokens, embeddings)
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")
    
    def _greedy_clustering(self, tokens: List[str], embeddings: torch.Tensor) -> List[ClusterInfo]:
        """
        Greedy clustering implementation following Algorithm 1.
        
        Sequentially forms spans by comparing adjacent tokens with similarity threshold.
        """
        clusters = []
        n = len(tokens)
        visited = set()
        
        t = 0
        while t < n:
            if t in visited:
                t += 1
                continue
                
            # Start new cluster with current token
            current_cluster = [t]
            visited.add(t)
            
            # Greedily add similar tokens
            for j in range(t + 1, n):
                if j in visited:
                    continue
                    
                # Calculate similarity between current token and candidate
                similarity = self._compute_cosine_similarity(
                    embeddings[t], embeddings[j]
                )
                
                if (similarity > self.similarity_threshold and 
                    len(current_cluster) < self.max_cluster_size):
                    current_cluster.append(j)
                    visited.add(j)
                else:
                    break
            
            # Create cluster if it meets minimum size requirement
            if len(current_cluster) >= self.min_cluster_size:
                cluster_info = self._create_cluster_info(
                    current_cluster, tokens, embeddings
                )
                if cluster_info.coherence_score >= self.coherence_threshold:
                    clusters.append(cluster_info)
            
            t += 1
        
        logger.debug(f"Greedy clustering formed {len(clusters)} clusters from {n} tokens")
        return clusters
    
    def _agglomerative_clustering(self, tokens: List[str], 
                                 embeddings: torch.Tensor) -> List[ClusterInfo]:
        """
        Agglomerative clustering using scikit-learn.
        
        Uses bottom-up hierarchical clustering with cosine similarity.
        """
        if len(tokens) < self.min_cluster_size:
            return []
        
        # Convert embeddings to numpy for sklearn
        embeddings_np = embeddings.cpu().numpy()
        
        # Perform agglomerative clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1.0 - self.similarity_threshold,  # Convert similarity to distance
            linkage='average',
            metric='cosine'
        )
        
        cluster_labels = clustering.fit_predict(embeddings_np)
        
        # Group tokens by cluster labels
        clusters = []
        unique_labels = np.unique(cluster_labels)
        
        for label in unique_labels:
            cluster_indices = np.where(cluster_labels == label)[0].tolist()
            
            if (len(cluster_indices) >= self.min_cluster_size and 
                len(cluster_indices) <= self.max_cluster_size):
                
                cluster_info = self._create_cluster_info(
                    cluster_indices, tokens, embeddings
                )
                if cluster_info.coherence_score >= self.coherence_threshold:
                    clusters.append(cluster_info)
        
        logger.debug(f"Agglomerative clustering formed {len(clusters)} clusters")
        return clusters
    
    def _dbscan_clustering(self, tokens: List[str], embeddings: torch.Tensor) -> List[ClusterInfo]:
        """
        DBSCAN clustering for density-based token grouping.
        
        Good for finding clusters of varying sizes and handling noise.
        """
        if len(tokens) < self.min_cluster_size:
            return []
        
        embeddings_np = embeddings.cpu().numpy()
        
        # DBSCAN with cosine distance
        clustering = DBSCAN(
            eps=1.0 - self.similarity_threshold,
            min_samples=self.min_cluster_size,
            metric='cosine'
        )
        
        cluster_labels = clustering.fit_predict(embeddings_np)
        
        clusters = []
        unique_labels = np.unique(cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
                
            cluster_indices = np.where(cluster_labels == label)[0].tolist()
            
            if len(cluster_indices) <= self.max_cluster_size:
                cluster_info = self._create_cluster_info(
                    cluster_indices, tokens, embeddings
                )
                if cluster_info.coherence_score >= self.coherence_threshold:
                    clusters.append(cluster_info)
        
        logger.debug(f"DBSCAN clustering formed {len(clusters)} clusters")
        return clusters
    
    def _create_cluster_info(self, indices: List[int], tokens: List[str], 
                           embeddings: torch.Tensor) -> ClusterInfo:
        """Create ClusterInfo object from cluster indices."""
        cluster_tokens = [tokens[i] for i in indices]
        cluster_embeddings = embeddings[indices]
        
        # Compute centroid
        centroid = cluster_embeddings.mean(dim=0)
        
        # Compute coherence score (average pairwise similarity)
        coherence_score = self._compute_cluster_coherence(cluster_embeddings)
        
        return ClusterInfo(
            indices=sorted(indices),
            tokens=cluster_tokens,
            centroid=centroid,
            coherence_score=coherence_score,
            size=len(indices),
            span_start=min(indices),
            span_end=max(indices)
        )
    
    def _compute_cosine_similarity(self, embedding1: torch.Tensor, 
                                  embedding2: torch.Tensor) -> float:
        """Compute cosine similarity between two embeddings."""
        return torch.cosine_similarity(
            embedding1.unsqueeze(0), embedding2.unsqueeze(0)
        ).item()
    
    def _compute_cluster_coherence(self, embeddings: torch.Tensor) -> float:
        """
        Compute internal coherence of a cluster.
        
        Uses average pairwise cosine similarity as coherence measure.
        """
        if embeddings.size(0) <= 1:
            return 1.0
        
        # Normalize embeddings
        normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Compute pairwise similarities
        similarity_matrix = torch.mm(normalized, normalized.t())
        
        # Get upper triangular part (excluding diagonal)
        n = similarity_matrix.size(0)
        upper_tri_mask = torch.triu(torch.ones(n, n), diagonal=1).bool()
        similarities = similarity_matrix[upper_tri_mask]
        
        return similarities.mean().item()
    
    def merge_overlapping_clusters(self, clusters: List[ClusterInfo]) -> List[ClusterInfo]:
        """
        Merge clusters that have overlapping token indices.
        
        Args:
            clusters: List of ClusterInfo objects
            
        Returns:
            List of merged clusters with no overlaps
        """
        if not clusters:
            return clusters
        
        # Sort clusters by start position
        sorted_clusters = sorted(clusters, key=lambda c: c.span_start)
        merged_clusters = []
        
        current_cluster = sorted_clusters[0]
        
        for next_cluster in sorted_clusters[1:]:
            # Check for overlap
            if (current_cluster.span_end >= next_cluster.span_start and
                self._should_merge_clusters(current_cluster, next_cluster)):
                
                # Merge clusters
                current_cluster = self._merge_two_clusters(current_cluster, next_cluster)
            else:
                merged_clusters.append(current_cluster)
                current_cluster = next_cluster
        
        merged_clusters.append(current_cluster)
        
        logger.debug(f"Merged {len(clusters)} clusters into {len(merged_clusters)}")
        return merged_clusters
    
    def _should_merge_clusters(self, cluster1: ClusterInfo, cluster2: ClusterInfo) -> bool:
        """Determine if two clusters should be merged."""
        # Check centroid similarity
        centroid_similarity = self._compute_cosine_similarity(
            cluster1.centroid, cluster2.centroid
        )
        
        return centroid_similarity > self.similarity_threshold
    
    def _merge_two_clusters(self, cluster1: ClusterInfo, cluster2: ClusterInfo) -> ClusterInfo:
        """Merge two clusters into one."""
        merged_indices = sorted(set(cluster1.indices + cluster2.indices))
        merged_tokens = cluster1.tokens + cluster2.tokens
        
        # Compute new centroid (weighted by cluster sizes)
        w1 = cluster1.size / (cluster1.size + cluster2.size)
        w2 = cluster2.size / (cluster1.size + cluster2.size)
        merged_centroid = w1 * cluster1.centroid + w2 * cluster2.centroid
        
        # Compute new coherence (average of both clusters)
        merged_coherence = (cluster1.coherence_score + cluster2.coherence_score) / 2
        
        return ClusterInfo(
            indices=merged_indices,
            tokens=merged_tokens,
            centroid=merged_centroid,
            coherence_score=merged_coherence,
            size=len(merged_indices),
            span_start=min(merged_indices),
            span_end=max(merged_indices)
        )
    
    def filter_clusters_by_quality(self, clusters: List[ClusterInfo], 
                                  min_coherence: float = None) -> List[ClusterInfo]:
        """
        Filter clusters based on quality metrics.
        
        Args:
            clusters: List of clusters to filter
            min_coherence: Minimum coherence threshold (uses default if None)
            
        Returns:
            Filtered list of high-quality clusters
        """
        if min_coherence is None:
            min_coherence = self.coherence_threshold
        
        filtered_clusters = []
        
        for cluster in clusters:
            # Check coherence threshold
            if cluster.coherence_score < min_coherence:
                continue
            
            # Check size constraints
            if (cluster.size < self.min_cluster_size or 
                cluster.size > self.max_cluster_size):
                continue
            
            filtered_clusters.append(cluster)
        
        logger.debug(f"Filtered {len(clusters)} clusters to {len(filtered_clusters)} high-quality ones")
        return filtered_clusters
    
    def get_clustering_stats(self, clusters: List[ClusterInfo]) -> Dict:
        """Get statistics about clustering results."""
        if not clusters:
            return {
                "num_clusters": 0,
                "total_tokens": 0,
                "avg_cluster_size": 0,
                "avg_coherence": 0,
                "compression_ratio": 0
            }
        
        total_tokens = sum(c.size for c in clusters)
        avg_cluster_size = total_tokens / len(clusters)
        avg_coherence = sum(c.coherence_score for c in clusters) / len(clusters)
        
        # Estimate compression ratio (assuming each cluster becomes 1 token)
        compression_ratio = len(clusters) / total_tokens if total_tokens > 0 else 0
        
        return {
            "num_clusters": len(clusters),
            "total_tokens": total_tokens,
            "avg_cluster_size": avg_cluster_size,
            "avg_coherence": avg_coherence,
            "compression_ratio": compression_ratio,
            "size_distribution": self._get_size_distribution(clusters),
            "coherence_distribution": self._get_coherence_distribution(clusters)
        }
    
    def _get_size_distribution(self, clusters: List[ClusterInfo]) -> Dict[int, int]:
        """Get distribution of cluster sizes."""
        size_counts = {}
        for cluster in clusters:
            size_counts[cluster.size] = size_counts.get(cluster.size, 0) + 1
        return size_counts
    
    def _get_coherence_distribution(self, clusters: List[ClusterInfo]) -> Dict[str, float]:
        """Get statistics about coherence scores."""
        if not clusters:
            return {"min": 0, "max": 0, "mean": 0, "std": 0}
        
        coherence_scores = [c.coherence_score for c in clusters]
        
        return {
            "min": min(coherence_scores),
            "max": max(coherence_scores),
            "mean": sum(coherence_scores) / len(coherence_scores),
            "std": np.std(coherence_scores)
        }


class AdaptiveCluster(SemanticCluster):
    """
    Adaptive clustering that adjusts parameters based on input characteristics.
    
    Dynamically modifies similarity thresholds and cluster sizes based on:
    - Text domain (technical vs. narrative)
    - Sequence length
    - Embedding distribution
    """
    
    def __init__(self, *args, adaptation_strategy: str = "entropy_based", **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptation_strategy = adaptation_strategy
        self.original_threshold = self.similarity_threshold
    
    def cluster_tokens(self, tokens: List[str], embeddings: torch.Tensor) -> List[ClusterInfo]:
        """Adaptive clustering with dynamic parameter adjustment."""
        
        # Analyze input characteristics
        analysis = self._analyze_input(tokens, embeddings)
        
        # Adapt parameters based on analysis
        self._adapt_parameters(analysis)
        
        # Perform clustering with adapted parameters
        clusters = super().cluster_tokens(tokens, embeddings)
        
        # Reset parameters
        self.similarity_threshold = self.original_threshold
        
        return clusters
    
    def _analyze_input(self, tokens: List[str], embeddings: torch.Tensor) -> Dict:
        """Analyze input characteristics for parameter adaptation."""
        
        # Compute embedding statistics
        embedding_mean = embeddings.mean(dim=0)
        embedding_std = embeddings.std(dim=0).mean().item()
        
        # Compute pairwise similarity statistics
        normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.mm(normalized, normalized.t())
        
        # Get upper triangular similarities (excluding diagonal)
        n = similarity_matrix.size(0)
        upper_tri_mask = torch.triu(torch.ones(n, n), diagonal=1).bool()
        similarities = similarity_matrix[upper_tri_mask]
        
        similarity_mean = similarities.mean().item()
        similarity_std = similarities.std().item()
        
        return {
            "sequence_length": len(tokens),
            "embedding_std": embedding_std,
            "similarity_mean": similarity_mean,
            "similarity_std": similarity_std,
            "avg_token_length": sum(len(token) for token in tokens) / len(tokens)
        }
    
    def _adapt_parameters(self, analysis: Dict):
        """Adapt clustering parameters based on input analysis."""
        
        if self.adaptation_strategy == "entropy_based":
            # Lower threshold for high-entropy (diverse) sequences
            if analysis["similarity_std"] > 0.2:
                self.similarity_threshold *= 0.9
            
            # Adjust cluster size based on sequence length
            if analysis["sequence_length"] > 1000:
                self.max_cluster_size = min(15, self.max_cluster_size + 3)
        
        elif self.adaptation_strategy == "domain_based":
            # Technical text (longer avg token length) - stricter clustering
            if analysis["avg_token_length"] > 6:
                self.similarity_threshold *= 1.1
            
            # Conversational text (shorter tokens) - more lenient
            elif analysis["avg_token_length"] < 4:
                self.similarity_threshold *= 0.95
        
        logger.debug(f"Adapted similarity threshold to {self.similarity_threshold:.3f}")


# Utility functions for histogram-based clustering
def histogram_clustering(embeddings: torch.Tensor, num_bins: int = 50, 
                        threshold: float = 0.7) -> List[List[int]]:
    """
    Fast clustering using histogram binning on cosine scores.
    
    Efficient alternative to full pairwise similarity computation.
    
    Args:
        embeddings: Token embeddings
        num_bins: Number of histogram bins
        threshold: Similarity threshold for clustering
        
    Returns:
        List of token index clusters
    """
    n = embeddings.size(0)
    if n <= 1:
        return [[i] for i in range(n)]
    
    # Normalize embeddings
    normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    # Create histogram bins based on first principal component
    U, S, V = torch.pca_lowrank(normalized, q=1)
    projections = torch.mm(normalized, V).squeeze()
    
    # Bin the projections
    min_proj, max_proj = projections.min(), projections.max()
    bin_size = (max_proj - min_proj) / num_bins
    
    bins = {}
    for i, proj in enumerate(projections):
        bin_idx = int((proj - min_proj) / bin_size)
        bin_idx = min(bin_idx, num_bins - 1)  # Handle edge case
        
        if bin_idx not in bins:
            bins[bin_idx] = []
        bins[bin_idx].append(i)
    
    # Form clusters from bins with sufficient similarity
    clusters = []
    for bin_tokens in bins.values():
        if len(bin_tokens) >= 2:
            # Verify similarity within bin
            bin_embeddings = normalized[bin_tokens]
            avg_similarity = torch.mm(bin_embeddings, bin_embeddings.t()).mean().item()
            
            if avg_similarity >= threshold:
                clusters.append(bin_tokens)
    
    return clusters
