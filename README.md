# SemToken: Semantic-Aware Tokenization for Efficient Long-Context Language Modeling

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2508.15190-b31b1b.svg)](https://arxiv.org/abs/2508.15190)

SemToken is a semantic-aware tokenization framework that dynamically adjusts token granularity based on local semantic density, achieving significant compression and speedup for long-context language modeling without sacrificing quality.

## Key Features

- **Semantic-Aware Compression**: Intelligently merges semantically equivalent tokens while preserving information-rich content
- **Adaptive Granularity**: Allocates fine-grained tokens to content-rich regions and coarse-grained tokens to repetitive spans
- **Query-Aware Processing**: Supports query-conditioned compression for better relevance preservation
- **Model-Agnostic**: Compatible with existing language models and attention mechanisms
- **Efficient Implementation**: Lightweight processing with linear complexity

## Performance

SemToken achieves impressive results across multiple benchmarks:

- **Up to 2.4× token reduction** with minimal quality degradation
- **1.9× speedup** in end-to-end inference latency
- **62% KV cache memory reduction**
- **Compatible** with FlashAttention and other acceleration methods

## Installation

```bash
# Install from source
git clone https://github.com/dongliu/SemToken.git
cd SemToken
pip install -e .

# Install dependencies
pip install torch transformers sentence-transformers scikit-learn numpy matplotlib seaborn tqdm
```

## Quick Start

### Basic Usage

```python
from semtoken import SemToken

# Initialize SemToken with default configuration
semtoken = SemToken()

# Your long text
text = """
Natural language processing is a field of artificial intelligence.
Natural language processing involves computational linguistics.
Machine learning models require large amounts of training data.
Machine learning algorithms learn patterns from data.
"""

# Perform semantic-aware tokenization
result = semtoken.tokenize(text)

print(f"Original tokens: {result['compression_stats']['original_count']}")
print(f"Compressed tokens: {result['compression_stats']['compressed_count']}")
print(f"Compression ratio: {result['compression_stats']['compression_ratio']:.2%}")
print(f"Theoretical speedup: {result['compression_stats']['theoretical_speedup']:.2f}x")
```

### Advanced Configuration

```python
from semtoken import SemToken, SemTokenConfig

# Custom configuration
config = SemTokenConfig(
    compression_ratio=0.4,          # Target 40% of original tokens
    similarity_threshold=0.8,       # Higher threshold for stricter clustering
    entropy_threshold=0.6,          # Higher threshold for fine-grained content
    encoder_model="sentence-transformers/all-MiniLM-L6-v2"
)

semtoken = SemToken(config=config)
result = semtoken.tokenize(text)
```

### Query-Aware Compression

```python
# Compress with query awareness
document = "Long document about AI and machine learning..."
query = "What is deep learning?"

result = semtoken.tokenize(document, query=query)
# Compression will preserve content more relevant to the query
```

### Batch Processing

```python
documents = [
    "Document 1 about artificial intelligence...",
    "Document 2 about machine learning...",
    "Document 3 about natural language processing..."
]

batch_results = semtoken.batch_process(documents)
```

## Architecture

SemToken operates in three main stages:

### 1. Semantic Embedding
- Extracts contextual embeddings using lightweight encoders (SimCSE, distilled BERT)
- Computes semantic fingerprints with sliding window context

### 2. Local Clustering  
- Groups semantically equivalent tokens using cosine similarity
- Supports multiple clustering algorithms (greedy, agglomerative, DBSCAN)
- Applies coherence filtering for quality control

### 3. Granularity Assignment
- Computes semantic entropy: `H(T) = Tr(Cov({fθ(xi) | xi ∈ T}))`
- Assigns adaptive granularity based on information density
- Balances compression ratios across content types

### 4. Budget Allocation
- Solves optimization: `max Σ H(x'i)` subject to budget constraints
- Supports query-aware importance scoring
- Provides autoregressive merging for generation tasks

## Algorithm

The core algorithm follows this process:

```python
def semtoken_algorithm(tokens, encoder, similarity_threshold=0.7, 
                      entropy_threshold=0.5, budget=None):
    # Step 1: Extract semantic fingerprints
    embeddings = [encoder.encode(context(token)) for token in tokens]
    
    # Step 2: Form spans via local similarity
    spans = []
    for i, token in enumerate(tokens):
        span = [i]
        for j in range(i+1, len(tokens)):
            if cosine_similarity(embeddings[i], embeddings[j]) > similarity_threshold:
                span.append(j)
            else:
                break
        spans.append(span)
    
    # Step 3: Compute semantic entropy
    entropies = [trace(cov(span_embeddings)) for span_embeddings in spans]
    
    # Step 4: Select top-B highest entropy spans
    selected_spans = top_k_by_entropy(spans, entropies, budget)
    
    return merge_tokens(selected_spans)
```

## Evaluation

Run comprehensive benchmarks:

```python
from evaluation.benchmark import run_paper_benchmark

# Run full benchmark suite
results, report, comparison = run_paper_benchmark()
print(report)
```

### Benchmark Results

| Method | Compression Ratio | Latency (ms/token) | Memory (GB) | PPL | F1 |
|--------|-------------------|-------------------|-------------|-----|-----|
| BPE (Default) | 0% | 61.2 | 4.1 | 17.3 | 59.4 |
| Entropy-Pruned | 25% | 48.4 | 2.9 | 18.2 | 57.8 |
| VQ-Tok | 33% | 47.9 | 2.8 | 18.0 | 58.2 |
| **SemToken** | **59%** | **30.4** | **1.5** | **17.0** | **59.9** |

## Use Cases

### Long Document Processing
- Legal document analysis
- Scientific paper comprehension  
- Book summarization

### Conversational AI
- Multi-turn dialogue systems
- Context-aware chatbots
- Memory-efficient inference

### Retrieval-Augmented Generation
- Efficient context compression
- Query-aware document selection
- Scalable knowledge bases

## Examples

Explore comprehensive examples:

```bash
# Basic usage examples
python examples/basic_usage.py

# Advanced configuration
python examples/advanced_config.py

# Integration with existing models
python examples/model_integration.py

# Performance benchmarking
python examples/benchmark_demo.py
```

## Configuration Options

### SemTokenConfig Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `compression_ratio` | Target compression ratio | 0.5 |
| `similarity_threshold` | Token similarity threshold (τ) | 0.7 |
| `entropy_threshold` | Semantic entropy threshold (δ) | 0.5 |
| `encoder_model` | Semantic encoder model | "all-MiniLM-L6-v2" |
| `context_window` | Context window size | 5 |
| `min_cluster_size` | Minimum cluster size | 2 |
| `max_cluster_size` | Maximum cluster size | 10 |
| `batch_size` | Processing batch size | 32 |
| `cache_embeddings` | Enable embedding cache | True |

### Preset Configurations

```python
# Conservative compression (high quality)
conservative_config = SemTokenConfig(
    compression_ratio=0.7,
    similarity_threshold=0.6,
    entropy_threshold=0.4
)

# Aggressive compression (high efficiency)  
aggressive_config = SemTokenConfig(
    compression_ratio=0.3,
    similarity_threshold=0.8,
    entropy_threshold=0.6
)
```

## Performance Tips

### Memory Optimization
- Use `cache_embeddings=True` for repeated processing
- Set appropriate `batch_size` based on GPU memory
- Consider using quantized models for the encoder

### Speed Optimization
- Enable `parallel_processing=True`
- Use smaller context windows for faster processing
- Cache frequently used embeddings

### Quality Optimization
- Increase `similarity_threshold` for stricter clustering
- Use larger `context_window` for better semantic understanding
- Fine-tune `entropy_threshold` for your domain

## Integration

### With Transformers

```python
from transformers import AutoModel, AutoTokenizer
from semtoken import SemToken

# Use with any transformer model
base_tokenizer = AutoTokenizer.from_pretrained("gpt2")
semtoken = SemToken(tokenizer=base_tokenizer)

# Compress input before model processing
text = "Your long input text..."
compressed = semtoken.tokenize(text)
```

### With FlashAttention

```python
# SemToken works seamlessly with attention optimizations
# Compression benefits stack with attention acceleration
# Expected combined speedup: 2.7× (SemToken) × 1.6× (FlashAttention) = 4.3×
```

## Citation

If you use SemToken in your research, please cite our paper:

```bibtex
@article{liu2025semtoken,
  title={SemToken: Semantic-Aware Tokenization for Efficient Long-Context Language Modeling},
  author={Liu, Dong and Yu, Yanxuan},
  journal={arXiv preprint arXiv:2508.15190},
  year={2025}
}
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/dongliu/SemToken.git
cd SemToken

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 semtoken/
black semtoken/
```

## License

This project is licensed under the MIT License。

## Acknowledgments

- Thanks to the Sentence-BERT team for semantic embedding models
- Inspired by recent work on efficient attention mechanisms
- Built on top of the excellent Transformers library

---

**SemToken** - Making long-context language modeling more efficient through semantic awareness!
