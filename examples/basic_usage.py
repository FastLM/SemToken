"""
Basic usage examples for SemToken.
"""

import torch
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from semtoken import SemToken, SemTokenConfig
from semtoken.utils import PerformanceProfiler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def basic_compression_example():
    """Basic example of semantic-aware tokenization."""
    
    print("=== Basic SemToken Compression Example ===\n")
    
    # Sample text with redundancy
    text = """
    The quick brown fox jumps over the lazy dog. The quick brown fox is very agile.
    Natural language processing is a field of artificial intelligence. 
    Natural language processing involves computational linguistics.
    Machine learning models require large amounts of training data.
    Machine learning algorithms learn patterns from data.
    The weather today is sunny and warm. The weather forecast predicts rain tomorrow.
    """
    
    # Initialize SemToken with default configuration
    semtoken = SemToken()
    
    print(f"Original text length: {len(text)} characters")
    print(f"Original text:\n{text}\n")
    
    # Perform semantic compression
    result = semtoken.tokenize(text)
    
    # Display results
    print("Compression Results:")
    print(f"- Original tokens: {result['compression_stats']['original_count']}")
    print(f"- Compressed tokens: {result['compression_stats']['compressed_count']}")
    print(f"- Compression ratio: {result['compression_stats']['compression_ratio']:.2%}")
    print(f"- Tokens saved: {result['compression_stats']['tokens_saved']}")
    print(f"- Theoretical speedup: {result['compression_stats']['theoretical_speedup']:.2f}x")
    print(f"- Memory reduction: {result['compression_stats']['memory_reduction']:.2%}")
    
    print(f"\nSemantic Analysis:")
    print(f"- Clusters formed: {result['semantic_info']['clusters_formed']}")
    print(f"- Semantic stats: {result['semantic_info']['semantic_stats']}")
    
    # Decode compressed tokens
    decoded_text = semtoken.decode(
        result['compressed_tokens'], 
        result['token_spans']
    )
    
    print(f"\nDecoded text:\n{decoded_text}")
    
    return result


def configuration_comparison():
    """Compare different SemToken configurations."""
    
    print("\n=== Configuration Comparison ===\n")
    
    # Sample long text
    text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    in contrast to the natural intelligence displayed by humans and animals. 
    Leading AI textbooks define the field as the study of "intelligent agents": 
    any device that perceives its environment and takes actions that maximize 
    its chance of successfully achieving its goals. Colloquially, the term 
    "artificial intelligence" is often used to describe machines that mimic 
    "cognitive" functions that humans associate with the human mind, such as 
    "learning" and "problem solving". As machines become increasingly capable, 
    tasks considered to require "intelligence" are often removed from the 
    definition of AI, a phenomenon known as the AI effect. A quip in Tesler's 
    Theorem says "AI is whatever hasn't been done yet." For instance, optical 
    character recognition is frequently excluded from things considered to be AI, 
    having become a routine technology. Modern machine learning techniques are 
    at the core of AI. Problems for AI applications include reasoning, knowledge 
    representation, planning, learning, natural language processing, perception 
    and the ability to move and manipulate objects.
    """ * 3  # Repeat for longer text
    
    configs = {
        "Conservative": SemTokenConfig(
            compression_ratio=0.8,
            similarity_threshold=0.6,
            entropy_threshold=0.4
        ),
        "Balanced": SemTokenConfig(
            compression_ratio=0.5,
            similarity_threshold=0.7,
            entropy_threshold=0.5
        ),
        "Aggressive": SemTokenConfig(
            compression_ratio=0.3,
            similarity_threshold=0.8,
            entropy_threshold=0.6
        )
    }
    
    results = {}
    
    for config_name, config in configs.items():
        print(f"Testing {config_name} configuration...")
        
        semtoken = SemToken(config=config)
        result = semtoken.tokenize(text)
        
        results[config_name] = result
        
        print(f"  - Compression ratio: {result['compression_stats']['compression_ratio']:.2%}")
        print(f"  - Theoretical speedup: {result['compression_stats']['theoretical_speedup']:.2f}x")
        print(f"  - Clusters formed: {result['semantic_info']['clusters_formed']}")
        print()
    
    return results


def query_aware_compression():
    """Demonstrate query-aware compression."""
    
    print("=== Query-Aware Compression ===\n")
    
    # Long document
    document = """
    Machine learning is a method of data analysis that automates analytical 
    model building. It is a branch of artificial intelligence based on the 
    idea that systems can learn from data, identify patterns and make decisions 
    with minimal human intervention. Deep learning is part of a broader family 
    of machine learning methods based on artificial neural networks with 
    representation learning. Learning can be supervised, semi-supervised or 
    unsupervised. Deep learning architectures such as deep neural networks, 
    deep belief networks, recurrent neural networks and convolutional neural 
    networks have been applied to fields including computer vision, speech 
    recognition, natural language processing, machine translation, bioinformatics 
    and drug design, where they have produced results comparable to and in some 
    cases surpassing human expert performance. Natural language processing (NLP) 
    is a subfield of linguistics, computer science, and artificial intelligence 
    concerned with the interactions between computers and human language, in 
    particular how to program computers to process and analyze large amounts 
    of natural language data. The result is a computer capable of "understanding" 
    the contents of documents, including the contextual nuances of the language 
    within them. The technology can then accurately extract information and 
    insights contained in the documents as well as categorize and organize the 
    documents themselves.
    """
    
    # Different queries
    queries = [
        "What is deep learning?",
        "How does natural language processing work?",
        "What are the applications of machine learning?"
    ]
    
    semtoken = SemToken()
    
    for query in queries:
        print(f"Query: {query}")
        
        # Compress with query awareness
        result = semtoken.tokenize(document, query=query)
        
        print(f"  - Compression ratio: {result['compression_stats']['compression_ratio']:.2%}")
        print(f"  - Clusters formed: {result['semantic_info']['clusters_formed']}")
        print(f"  - Processing time: {result['metadata']['processing_time']:.3f}s")
        print()
    
    return results


def performance_profiling():
    """Demonstrate performance profiling capabilities."""
    
    print("=== Performance Profiling ===\n")
    
    # Create test texts of different lengths
    base_text = "The field of artificial intelligence continues to evolve rapidly. "
    test_texts = {
        "Short": base_text * 10,
        "Medium": base_text * 50,
        "Long": base_text * 200
    }
    
    profiler = PerformanceProfiler()
    semtoken = SemToken()
    
    for length_name, text in test_texts.items():
        print(f"Profiling {length_name} text ({len(text)} characters)...")
        
        # Profile the compression
        profiler.start_profile(f"compress_{length_name}")
        result = semtoken.tokenize(text)
        profile_result = profiler.end_profile()
        
        print(f"  - Processing time: {profile_result['total_time']:.3f}s")
        print(f"  - Memory delta: {profile_result['memory_delta']:.2f}MB")
        print(f"  - Compression ratio: {result['compression_stats']['compression_ratio']:.2%}")
        print()
    
    # Get overall performance summary
    summary = profiler.get_performance_summary()
    print("Performance Summary:")
    for operation, stats in summary.items():
        print(f"  {operation}:")
        print(f"    - Average time: {stats['avg_time']:.3f}s")
        print(f"    - Total runs: {stats['count']}")
        print(f"    - Total time: {stats['total_time']:.3f}s")
    
    return summary


def batch_processing_example():
    """Demonstrate batch processing capabilities."""
    
    print("\n=== Batch Processing Example ===\n")
    
    # Multiple documents
    documents = [
        "The history of artificial intelligence began in antiquity with myths and stories.",
        "Machine learning algorithms can be categorized into supervised and unsupervised learning.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning uses neural networks with multiple layers to learn complex patterns.",
        "Computer vision allows machines to interpret and understand visual information."
    ]
    
    semtoken = SemToken()
    
    # Process batch
    print("Processing batch of documents...")
    batch_results = semtoken.batch_process(documents)
    
    print(f"Processed {len(batch_results)} documents")
    print()
    
    for i, result in enumerate(batch_results):
        print(f"Document {i+1}:")
        print(f"  - Original tokens: {result['compression_stats']['original_count']}")
        print(f"  - Compressed tokens: {result['compression_stats']['compressed_count']}")
        print(f"  - Compression ratio: {result['compression_stats']['compression_ratio']:.2%}")
        print()
    
    # Calculate batch statistics
    total_original = sum(r['compression_stats']['original_count'] for r in batch_results)
    total_compressed = sum(r['compression_stats']['compressed_count'] for r in batch_results)
    overall_compression = total_compressed / total_original
    
    print(f"Batch Summary:")
    print(f"  - Total original tokens: {total_original}")
    print(f"  - Total compressed tokens: {total_compressed}")
    print(f"  - Overall compression ratio: {overall_compression:.2%}")
    
    return batch_results


def advanced_configuration():
    """Demonstrate advanced configuration options."""
    
    print("\n=== Advanced Configuration ===\n")
    
    # Create custom configuration
    custom_config = SemTokenConfig(
        # Encoder settings
        encoder_model="sentence-transformers/all-MiniLM-L6-v2",
        context_window=7,
        
        # Clustering parameters
        similarity_threshold=0.75,
        min_cluster_size=2,
        max_cluster_size=8,
        
        # Granularity settings
        entropy_threshold=0.55,
        density_window=12,
        
        # Budget allocation
        compression_ratio=0.4,
        priority_boost=1.3,
        
        # Performance settings
        batch_size=16,
        cache_embeddings=True,
        parallel_processing=True
    )
    
    text = """
    The development of transformer architectures has revolutionized natural language processing.
    Attention mechanisms allow models to focus on relevant parts of the input sequence.
    BERT, GPT, and other transformer-based models have achieved state-of-the-art results.
    Pre-training on large corpora followed by fine-tuning has become the standard approach.
    Self-attention computes representations by relating different positions in the sequence.
    Multi-head attention allows the model to attend to different types of information.
    The encoder-decoder architecture is particularly useful for sequence-to-sequence tasks.
    """
    
    # Compare with default configuration
    default_semtoken = SemToken()
    custom_semtoken = SemToken(config=custom_config)
    
    print("Comparing default vs custom configuration:")
    print()
    
    # Default configuration
    default_result = default_semtoken.tokenize(text)
    print("Default Configuration:")
    print(f"  - Compression ratio: {default_result['compression_stats']['compression_ratio']:.2%}")
    print(f"  - Clusters formed: {default_result['semantic_info']['clusters_formed']}")
    
    # Custom configuration
    custom_result = custom_semtoken.tokenize(text)
    print("Custom Configuration:")
    print(f"  - Compression ratio: {custom_result['compression_stats']['compression_ratio']:.2%}")
    print(f"  - Clusters formed: {custom_result['semantic_info']['clusters_formed']}")
    
    return default_result, custom_result


def main():
    """Run all examples."""
    
    print("SemToken Examples")
    print("=" * 50)
    
    try:
        # Basic compression
        basic_result = basic_compression_example()
        
        # Configuration comparison
        config_results = configuration_comparison()
        
        # Query-aware compression
        query_results = query_aware_compression()
        
        # Performance profiling
        perf_summary = performance_profiling()
        
        # Batch processing
        batch_results = batch_processing_example()
        
        # Advanced configuration
        advanced_results = advanced_configuration()
        
        print("\nAll examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise


if __name__ == "__main__":
    main()
