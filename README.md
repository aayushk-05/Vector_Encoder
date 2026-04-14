# NLP Vector Search Engine (C++)

A from-scratch C++ pipeline that encodes natural language text into vectors and retrieves relevant web results — sites, images, videos, news, and text — using semantic similarity search.

---

## Overview

This project implements a full NLP-to-retrieval pipeline in C++, covering tokenization, subword handling, vocabulary management, n-gram extraction, Markov chain modeling, TF-IDF scoring, vector encoding, and approximate nearest-neighbour (ANN) web retrieval — with no Python runtime dependency.

```
Raw text → Tokenizer → Vocabulary → N-grams + Markov → TF-IDF → Vector → ANN Search → Web Results
```

---

## Features

- **Tokenizer** — regex-based splitting with Unicode support via ICU; zero-copy slicing using `std::string_view`
- **Subword handler** — Byte Pair Encoding (BPE) with iterative merge rules; resolves out-of-vocabulary (OOV) tokens
- **Vocabulary** — bidirectional `word ↔ id` mapping with configurable vocab size and `[UNK]` fallback
- **N-gram extractor** — sliding window over token ID sequences; supports unigram, bigram, and trigram features
- **Markov chain** — bigram/trigram transition probability table for query expansion
- **TF-IDF engine** — sparse document-term matrix with log-IDF weighting; compatible with Eigen sparse vectors
- **Vector encoder** — dense `float32` embeddings via `Eigen::VectorXf`; SIMD-accelerated dot products with OpenBLAS
- **Retrieval layer** — ANN index via `hnswlib` or FAISS C API; HTTP queries via `libcurl`; JSON parsing via `nlohmann/json`

---

## Project Structure

```
nlp-vector-search/
├── src/
│   ├── tokenizer.hpp / .cpp       # Text splitting, Unicode, BPE
│   ├── vocabulary.hpp / .cpp      # word↔id maps, vocab building
│   ├── ngram.hpp / .cpp           # Sliding window n-gram extraction
│   ├── markov.hpp / .cpp          # Transition table, query expansion
│   ├── tfidf.hpp / .cpp           # TF-IDF scoring, sparse vectors
│   ├── encoder.hpp / .cpp         # Dense vector encoding, BLAS ops
│   ├── retriever.hpp / .cpp       # ANN search, HTTP, JSON parsing
│   └── main.cpp                   # Entry point
├── data/
│   ├── vocab.txt                  # Pre-built vocabulary file
│   └── bpe_merges.txt             # BPE merge rules
├── CMakeLists.txt
└── README.md
```

---

## Dependencies

| Library | Purpose | Install |
|---|---|---|
| [Eigen](https://eigen.tuxfamily.org/) | Linear algebra, dense/sparse vectors | `apt install libeigen3-dev` |
| [ICU](https://icu.unicode.org/) | Unicode tokenization | `apt install libicu-dev` |
| [hnswlib](https://github.com/nmslib/hnswlib) | ANN index (header-only) | Copy headers to `include/` |
| [libcurl](https://curl.se/libcurl/) | HTTP requests to search API | `apt install libcurl4-openssl-dev` |
| [nlohmann/json](https://github.com/nlohmann/json) | JSON response parsing | `apt install nlohmann-json3-dev` |
| [OpenBLAS](https://www.openblas.net/) | SIMD-accelerated dot products (optional) | `apt install libopenblas-dev` |

---

## Build

```bash
git clone https://github.com/yourname/nlp-vector-search.git
cd nlp-vector-search
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Requires CMake 3.15+ and a C++17-capable compiler (GCC 9+, Clang 10+).

---

## Usage

```bash
./nlp_search "best machine learning books 2024"
```

Output:

```
Query vector: [0.14, -0.83, 0.27, ...]  (512-dim)

Results:
[site]  https://example.com/ml-books     score: 0.94
[news]  https://news.example.com/ai      score: 0.91
[image] https://img.example.com/covers   score: 0.87
[video] https://video.example.com/ml     score: 0.85
```

### Programmatic API

```cpp
#include "tokenizer.hpp"
#include "encoder.hpp"
#include "retriever.hpp"

int main() {
    Tokenizer tok("data/vocab.txt", "data/bpe_merges.txt");
    Encoder enc(512);  // embedding dimension
    Retriever ret("https://api.yoursearchprovider.com", "<API_KEY>");

    auto tokens  = tok.tokenize("climate change renewable energy");
    auto ngrams  = NGram::extract(tokens, 2);          // bigrams
    auto tfidf   = TFIDF::score(tokens, ngrams);
    auto vec     = enc.encode(tfidf);
    auto results = ret.search(vec, /*top_k=*/10);

    for (auto& r : results)
        std::cout << r.type << "\t" << r.url << "\t" << r.score << "\n";
}
```

---

## Pipeline Design

### Tokenizer + Subword handling

Splits text on whitespace and punctuation using `std::regex`. Subword tokenization uses a pre-trained BPE vocabulary: unknown words are iteratively split into their largest known subword units until all pieces exist in the vocabulary. The merge table is stored as:

```cpp
std::unordered_map<std::pair<int,int>, int, PairHash> bpe_merges;
```

### Vocabulary

Bidirectional lookup with a configurable max size (default 32,000 tokens). Tokens beyond the vocabulary are mapped to `[UNK]` (id = 0).

```cpp
std::unordered_map<std::string, int> word2idx;
std::vector<std::string>             idx2word;
```

### N-gram extraction

A sliding `std::deque<int>` over the token ID stream generates all n-grams up to order n. Each n-gram is hashed to a `size_t` feature key using FNV-1a for fast lookup.

### Markov chain

Builds a transition probability table during corpus ingestion:

```cpp
std::map<std::vector<int>, std::map<int, float>> transitions;
```

At query time, the Markov model suggests likely next tokens given the query's tail, expanding coverage for the retrieval step.

### TF-IDF scoring

Sparse term-frequency vectors are stored as `std::unordered_map<int, float>`. IDF values are precomputed over a reference corpus and stored in a flat array indexed by token id. Final TF-IDF weights are computed as:

```
weight(t, d) = tf(t, d) × log(N / df(t) + 1)
```

### Vector encoding

TF-IDF weights are projected into a dense embedding space using a learned weight matrix (`Eigen::MatrixXf`). The result is an L2-normalized `float32` vector ready for cosine similarity search.

### Retrieval

The query vector is searched against a pre-built HNSW index (`hnswlib`) of document embeddings. Top-k results are returned with their cosine similarity scores. Result URLs are then enriched by calling a web search API endpoint via `libcurl`, and the JSON response is parsed with `nlohmann/json`.

---

## Configuration

Edit `config.hpp` or pass at runtime:

```cpp
struct Config {
    int    vocab_size    = 32000;
    int    ngram_order   = 2;       // 1=unigram, 2=bigram, 3=trigram
    int    markov_order  = 2;
    int    embed_dim     = 512;
    int    top_k         = 10;
    float  idf_smoothing = 1.0f;
    std::string api_key  = "";
    std::string api_url  = "";
};
```

---

## Roadmap

- [ ] BERT-style contextual embeddings via ONNX Runtime
- [ ] Persistent HNSW index with incremental updates
- [ ] Multi-language tokenizer (Bengali, Hindi, Arabic)
- [ ] gRPC server mode for use as a microservice
- [ ] Result re-ranking with cross-encoder scoring
- [ ] CLI flags and config file support

---

## License

MIT License. See `LICENSE` for details.
