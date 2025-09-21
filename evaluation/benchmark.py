"""
Comprehensive benchmarking suite for SemToken evaluation.
"""

import torch
import numpy as np
import time
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import sys
sys.path.append('..')
from semtoken import SemToken, SemTokenConfig
from semtoken.utils import PerformanceProfiler, compute_compression_metrics

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    dataset_name: str
    model_name: str
    config_name: str
    
    # Performance metrics
    compression_ratio: float
    inference_latency_ms: float
    memory_usage_mb: float
    throughput_tokens_per_sec: float
    
    # Quality metrics
    perplexity: Optional[float] = None
    f1_score: Optional[float] = None
    rouge_l: Optional[float] = None
    exact_match: Optional[float] = None
    
    # Efficiency metrics
    kv_cache_reduction: float = 0.0
    attention_ops_reduction: float = 0.0
    theoretical_speedup: float = 1.0
    
    # Additional metadata
    original_tokens: int = 0
    compressed_tokens: int = 0
    processing_time_ms: float = 0.0
    error_message: Optional[str] = None


class SemTokenBenchmark:
    """
    Comprehensive benchmark suite for evaluating SemToken performance.
    
    Tests across multiple datasets, models, and configurations to provide
    thorough evaluation of compression quality and efficiency.
    """
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """Initialize benchmark suite."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.profiler = PerformanceProfiler()
        self.results = []
        
        # Benchmark configurations
        self.configs = {
            "aggressive": SemTokenConfig(
                compression_ratio=0.3,
                similarity_threshold=0.8,
                entropy_threshold=0.6
            ),
            "balanced": SemTokenConfig(
                compression_ratio=0.5,
                similarity_threshold=0.7,
                entropy_threshold=0.5
            ),
            "conservative": SemTokenConfig(
                compression_ratio=0.7,
                similarity_threshold=0.6,
                entropy_threshold=0.4
            )
        }
        
        logger.info(f"Benchmark suite initialized, output dir: {self.output_dir}")
    
    def run_full_benchmark(self, datasets: List[str] = None, 
                          models: List[str] = None) -> List[BenchmarkResult]:
        """
        Run comprehensive benchmark across datasets and models.
        
        Args:
            datasets: List of dataset names to test
            models: List of model names to test
            
        Returns:
            List of benchmark results
        """
        if datasets is None:
            datasets = ["wikitext-103", "longbench", "booksum"]
        
        if models is None:
            models = ["gpt2", "llama-7b"]
        
        logger.info(f"Starting full benchmark: {len(datasets)} datasets, "
                   f"{len(models)} models, {len(self.configs)} configs")
        
        total_runs = len(datasets) * len(models) * len(self.configs)
        
        with tqdm(total=total_runs, desc="Benchmarking") as pbar:
            for dataset_name in datasets:
                for model_name in models:
                    for config_name, config in self.configs.items():
                        try:
                            result = self._run_single_benchmark(
                                dataset_name, model_name, config_name, config
                            )
                            self.results.append(result)
                            
                        except Exception as e:
                            logger.error(f"Benchmark failed: {dataset_name}, {model_name}, "
                                       f"{config_name}: {e}")
                            # Add failed result
                            failed_result = BenchmarkResult(
                                dataset_name=dataset_name,
                                model_name=model_name,
                                config_name=config_name,
                                compression_ratio=0.0,
                                inference_latency_ms=0.0,
                                memory_usage_mb=0.0,
                                throughput_tokens_per_sec=0.0,
                                error_message=str(e)
                            )
                            self.results.append(failed_result)
                        
                        pbar.update(1)
        
        # Save results
        self._save_results()
        
        logger.info(f"Benchmark completed: {len(self.results)} results")
        return self.results
    
    def _run_single_benchmark(self, dataset_name: str, model_name: str,
                             config_name: str, config: SemTokenConfig) -> BenchmarkResult:
        """Run benchmark for a single configuration."""
        
        logger.debug(f"Running benchmark: {dataset_name}, {model_name}, {config_name}")
        
        # Load test data
        test_texts = self._load_test_data(dataset_name)
        
        # Initialize SemToken
        semtoken = SemToken(config=config)
        
        # Performance metrics
        compression_ratios = []
        latencies = []
        memory_usages = []
        processing_times = []
        
        original_token_counts = []
        compressed_token_counts = []
        
        # Quality metrics (would need actual model evaluation)
        quality_metrics = {"perplexity": None, "f1": None, "rouge_l": None}
        
        # Run on sample of texts
        sample_size = min(10, len(test_texts))
        
        for i, text in enumerate(test_texts[:sample_size]):
            try:
                # Time the compression
                start_time = time.time()
                
                self.profiler.start_profile(f"compress_{i}")
                result = semtoken.tokenize(text)
                profile_result = self.profiler.end_profile()
                
                end_time = time.time()
                
                # Collect metrics
                compression_ratios.append(result['compression_stats']['compression_ratio'])
                latencies.append((end_time - start_time) * 1000)  # ms
                processing_times.append(profile_result.get('total_time', 0) * 1000)
                
                original_token_counts.append(result['compression_stats']['original_count'])
                compressed_token_counts.append(result['compression_stats']['compressed_count'])
                
                # Memory usage (simplified)
                memory_usages.append(profile_result.get('memory_delta', 0))
                
            except Exception as e:
                logger.warning(f"Failed to process text {i}: {e}")
                continue
        
        # Calculate aggregate metrics
        avg_compression_ratio = np.mean(compression_ratios) if compression_ratios else 0
        avg_latency = np.mean(latencies) if latencies else 0
        avg_memory = np.mean(memory_usages) if memory_usages else 0
        avg_processing_time = np.mean(processing_times) if processing_times else 0
        
        total_original = sum(original_token_counts)
        total_compressed = sum(compressed_token_counts)
        
        # Calculate throughput
        total_time_sec = sum(latencies) / 1000 if latencies else 1
        throughput = total_original / total_time_sec if total_time_sec > 0 else 0
        
        # Efficiency metrics
        kv_cache_reduction = 1.0 - avg_compression_ratio
        attention_ops_reduction = 1.0 - (avg_compression_ratio ** 2)
        theoretical_speedup = 1.0 / avg_compression_ratio if avg_compression_ratio > 0 else 1.0
        
        return BenchmarkResult(
            dataset_name=dataset_name,
            model_name=model_name,
            config_name=config_name,
            compression_ratio=avg_compression_ratio,
            inference_latency_ms=avg_latency,
            memory_usage_mb=avg_memory,
            throughput_tokens_per_sec=throughput,
            kv_cache_reduction=kv_cache_reduction,
            attention_ops_reduction=attention_ops_reduction,
            theoretical_speedup=theoretical_speedup,
            original_tokens=total_original,
            compressed_tokens=total_compressed,
            processing_time_ms=avg_processing_time,
            **quality_metrics
        )
    
    def _load_test_data(self, dataset_name: str) -> List[str]:
        """Load test data for the specified dataset."""
        
        # Mock data for demonstration - replace with actual dataset loading
        mock_data = {
            "wikitext-103": [
                "The quick brown fox jumps over the lazy dog. " * 50,
                "Natural language processing is a field of artificial intelligence. " * 30,
                "Machine learning models require large amounts of training data. " * 40
            ],
            "longbench": [
                "In this comprehensive analysis of modern computational linguistics, " * 100,
                "The theoretical foundations of semantic tokenization rest upon " * 80,
                "Experimental validation of compression algorithms demonstrates " * 90
            ],
            "booksum": [
                "Chapter 1: The protagonist begins their journey through the mystical forest. " * 60,
                "The narrative structure of this novel employs multiple perspectives. " * 70,
                "Literary analysis reveals complex thematic elements throughout. " * 50
            ]
        }
        
        return mock_data.get(dataset_name, mock_data["wikitext-103"])
    
    def _save_results(self):
        """Save benchmark results to files."""
        
        # Save as JSON
        results_data = []
        for result in self.results:
            result_dict = {
                "dataset_name": result.dataset_name,
                "model_name": result.model_name,
                "config_name": result.config_name,
                "compression_ratio": result.compression_ratio,
                "inference_latency_ms": result.inference_latency_ms,
                "memory_usage_mb": result.memory_usage_mb,
                "throughput_tokens_per_sec": result.throughput_tokens_per_sec,
                "kv_cache_reduction": result.kv_cache_reduction,
                "attention_ops_reduction": result.attention_ops_reduction,
                "theoretical_speedup": result.theoretical_speedup,
                "original_tokens": result.original_tokens,
                "compressed_tokens": result.compressed_tokens,
                "processing_time_ms": result.processing_time_ms,
                "error_message": result.error_message
            }
            results_data.append(result_dict)
        
        results_file = self.output_dir / "benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
    
    def generate_report(self) -> str:
        """Generate comprehensive benchmark report."""
        
        if not self.results:
            return "No benchmark results available."
        
        report = []
        report.append("# SemToken Benchmark Report\n")
        
        # Summary statistics
        successful_results = [r for r in self.results if r.error_message is None]
        failed_results = [r for r in self.results if r.error_message is not None]
        
        report.append(f"## Summary")
        report.append(f"- Total runs: {len(self.results)}")
        report.append(f"- Successful: {len(successful_results)}")
        report.append(f"- Failed: {len(failed_results)}")
        report.append("")
        
        if successful_results:
            # Overall performance
            avg_compression = np.mean([r.compression_ratio for r in successful_results])
            avg_latency = np.mean([r.inference_latency_ms for r in successful_results])
            avg_speedup = np.mean([r.theoretical_speedup for r in successful_results])
            
            report.append(f"## Overall Performance")
            report.append(f"- Average compression ratio: {avg_compression:.3f}")
            report.append(f"- Average inference latency: {avg_latency:.2f} ms")
            report.append(f"- Average theoretical speedup: {avg_speedup:.2f}x")
            report.append("")
            
            # Performance by configuration
            report.append(f"## Performance by Configuration")
            for config_name in self.configs.keys():
                config_results = [r for r in successful_results if r.config_name == config_name]
                if config_results:
                    config_compression = np.mean([r.compression_ratio for r in config_results])
                    config_speedup = np.mean([r.theoretical_speedup for r in config_results])
                    
                    report.append(f"### {config_name.title()}")
                    report.append(f"- Compression ratio: {config_compression:.3f}")
                    report.append(f"- Theoretical speedup: {config_speedup:.2f}x")
                    report.append("")
        
        # Error analysis
        if failed_results:
            report.append(f"## Failed Runs")
            for result in failed_results:
                report.append(f"- {result.dataset_name}, {result.model_name}, "
                            f"{result.config_name}: {result.error_message}")
            report.append("")
        
        return "\n".join(report)
    
    def create_visualizations(self):
        """Create visualization plots for benchmark results."""
        
        successful_results = [r for r in self.results if r.error_message is None]
        
        if not successful_results:
            logger.warning("No successful results to visualize")
            return
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Compression ratio by configuration
        config_names = [r.config_name for r in successful_results]
        compression_ratios = [r.compression_ratio for r in successful_results]
        
        axes[0, 0].bar(config_names, compression_ratios)
        axes[0, 0].set_title('Compression Ratio by Configuration')
        axes[0, 0].set_ylabel('Compression Ratio')
        
        # 2. Theoretical speedup by configuration
        speedups = [r.theoretical_speedup for r in successful_results]
        axes[0, 1].bar(config_names, speedups)
        axes[0, 1].set_title('Theoretical Speedup by Configuration')
        axes[0, 1].set_ylabel('Speedup (×)')
        
        # 3. Latency vs Compression scatter plot
        latencies = [r.inference_latency_ms for r in successful_results]
        axes[1, 0].scatter(compression_ratios, latencies, alpha=0.6)
        axes[1, 0].set_xlabel('Compression Ratio')
        axes[1, 0].set_ylabel('Inference Latency (ms)')
        axes[1, 0].set_title('Latency vs Compression Trade-off')
        
        # 4. Memory usage by configuration
        memory_usages = [r.memory_usage_mb for r in successful_results]
        axes[1, 1].bar(config_names, memory_usages)
        axes[1, 1].set_title('Memory Usage by Configuration')
        axes[1, 1].set_ylabel('Memory Usage (MB)')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "benchmark_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {plot_file}")
    
    def compare_with_baselines(self, baseline_results: Dict) -> Dict:
        """Compare SemToken results with baseline methods."""
        
        successful_results = [r for r in self.results if r.error_message is None]
        
        if not successful_results:
            return {"error": "No successful SemToken results to compare"}
        
        # Calculate SemToken averages
        semtoken_metrics = {
            "compression_ratio": np.mean([r.compression_ratio for r in successful_results]),
            "inference_latency_ms": np.mean([r.inference_latency_ms for r in successful_results]),
            "theoretical_speedup": np.mean([r.theoretical_speedup for r in successful_results]),
            "memory_reduction": np.mean([r.kv_cache_reduction for r in successful_results])
        }
        
        # Compare with baselines
        comparison = {"semtoken": semtoken_metrics}
        
        for baseline_name, baseline_metrics in baseline_results.items():
            comparison[baseline_name] = baseline_metrics
            
            # Calculate improvements
            improvement_key = f"improvement_over_{baseline_name}"
            comparison[improvement_key] = {}
            
            for metric, semtoken_value in semtoken_metrics.items():
                baseline_value = baseline_metrics.get(metric, 0)
                if baseline_value > 0:
                    if metric in ["compression_ratio", "inference_latency_ms"]:
                        # Lower is better
                        improvement = (baseline_value - semtoken_value) / baseline_value
                    else:
                        # Higher is better
                        improvement = (semtoken_value - baseline_value) / baseline_value
                    
                    comparison[improvement_key][metric] = improvement
        
        return comparison


def run_paper_benchmark():
    """Run the benchmark setup described in the paper."""
    
    benchmark = SemTokenBenchmark()
    
    # Paper datasets
    datasets = ["wikitext-103", "longbench", "booksum", "chartqa"]
    
    # Paper models
    models = ["llama-2-7b", "gpt-j-6b", "gpt-neox-20b"]
    
    # Run benchmark
    results = benchmark.run_full_benchmark(datasets, models)
    
    # Generate report
    report = benchmark.generate_report()
    
    # Save report
    report_file = benchmark.output_dir / "paper_benchmark_report.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    # Create visualizations
    benchmark.create_visualizations()
    
    # Compare with paper baselines
    paper_baselines = {
        "bpe_default": {
            "compression_ratio": 1.0,
            "inference_latency_ms": 61.2,
            "theoretical_speedup": 1.0,
            "memory_reduction": 0.0
        },
        "entropy_pruned": {
            "compression_ratio": 0.75,
            "inference_latency_ms": 48.4,
            "theoretical_speedup": 1.3,
            "memory_reduction": 0.29
        },
        "vq_tok": {
            "compression_ratio": 0.67,
            "inference_latency_ms": 47.9,
            "theoretical_speedup": 1.3,
            "memory_reduction": 0.32
        }
    }
    
    comparison = benchmark.compare_with_baselines(paper_baselines)
    
    # Save comparison
    comparison_file = benchmark.output_dir / "baseline_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    logger.info("Paper benchmark completed successfully")
    return results, report, comparison


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run paper benchmark
    results, report, comparison = run_paper_benchmark()
    
    print("Benchmark Report:")
    print("=" * 50)
    print(report)
    
    print("\nBaseline Comparison:")
    print("=" * 50)
    print(json.dumps(comparison, indent=2))
