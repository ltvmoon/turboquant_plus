# TurboQuant+

Implementation of [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) (ICLR 2026) — KV cache compression for local LLM inference, with planned extensions beyond the paper.

> **Why "Plus"?** The base TurboQuant paper is v1. We have ideas for improvements coming post-v1 — adaptive bit allocation, temporal decay compression, expert-aware MoE compression, and more. The "plus" is what comes next.

Compresses transformer KV cache **up to 4.9×** using PolarQuant + QJL. Paper claims zero accuracy loss at 3.5-bit; our prototype achieves cosine similarity 0.95 at 3.5-bit on real Qwen3 KV tensors.

**Working end-to-end on Apple Silicon** — Qwen 3.5 35B-A3B MoE generating coherent text with 3-bit TurboQuant KV cache on M5 Max via llama.cpp Metal.

## Status: v1 Complete

- 141 Python tests, 100% code coverage
- C port integrated into llama.cpp with Metal GPU kernels
- `--cache-type-k turbo3 --cache-type-v turbo3` works on Apple Silicon
- Compression ratios match [Prince Canuma's MLX results](https://x.com/Prince_Canuma): 2.5-bit=4.9×, 3.5-bit=3.8×
- Rotation Gaussianization validated on real Qwen3 KV tensors (kurtosis 900 → 2.9)

## Quick Start — Python Prototype

```bash
git clone https://github.com/TheTom/turboquant_plus.git
cd turboquant_plus
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run tests (141 tests, 100% coverage)
python3 -m pytest tests/ -v

# Run demo
python3 benchmarks/demo.py

# Compare with Prince Canuma's results
python3 benchmarks/test_outlier_comparison.py

# Validate with real model tensors (requires torch + transformers)
pip install transformers torch accelerate
python3 benchmarks/validate_real_model.py
```

## Quick Start — llama.cpp (Apple Silicon)

```bash
# Build llama.cpp with TurboQuant (from feature branch)
cd ~/local_llms/llama.cpp
git checkout feature/turboquant-kv-cache
cmake -B build -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Run with TurboQuant 3-bit KV cache (4.9× compression)
./build/bin/llama-server \
  -m models/your-model.gguf \
  -ngl 99 -c 262144 -fa on \
  --cache-type-k turbo3 --cache-type-v turbo3 \
  -np 1 --host 0.0.0.0 --port 8080

# Available cache types: turbo3 (3-bit, 4.9×), turbo4 (4-bit, 3.8×)
```

## Compression Results

| Config | Compression | Cosine Sim | MSE |
|--------|-------------|------------|-----|
| TurboQuant 2-bit | 7.1× | 0.79 | 0.0047 |
| TurboQuant 2.5-bit (outlier) | **4.9×** | 0.86 | 0.0029 |
| TurboQuant 3-bit | 4.9× | 0.91 | 0.0018 |
| TurboQuant 3.5-bit (outlier) | **3.8×** | 0.95 | 0.0009 |
| TurboQuant 4-bit | 3.8× | 0.96 | 0.0007 |


## Architecture

```
Input: KV cache vector x ∈ R^d (one attention head)
    │
    ├── Extract norm: γ = ||x||, x̂ = x/γ
    │
    ├── Stage 1: PolarQuant (b-1 bits)
    │   Random rotation Π → coordinates ~ N(0, 1/d)
    │   → optimal scalar quantization per coordinate
    │
    ├── Stage 2: QJL (1 bit)
    │   sign(S · residual) → unbiased inner product correction
    │
    └── Output: CompressedVector(indices, signs, norms)
        Total: b bits per coordinate
```

## Project Structure

```
turboquant/
├── rotation.py      # Random rotation matrices (dense QR + fast Walsh-Hadamard)
├── codebook.py      # Optimal centroid computation (closed-form + Lloyd's)
├── polar_quant.py   # PolarQuant (Algorithm 1) — with norm extraction
├── qjl.py           # QJL 1-bit quantizer
├── turboquant.py    # Full TurboQuant (Algorithm 2)
├── kv_cache.py      # KV cache integration layer
├── outlier.py       # Outlier channel strategy (2.5-bit, 3.5-bit)
└── utils.py         # Bit packing, memory measurement

tests/               # 141 tests, 100% coverage on core modules
benchmarks/
├── demo.py                    # Quick compression demo
├── test_with_llama.py         # Integration test at Qwen 3.5 dimensions
├── test_outlier_comparison.py # Comparison with Prince Canuma's results
└── validate_real_model.py     # Phase A: real model tensor validation
```

## Development Workflow

Every change follows this loop (see [CLAUDE.md](CLAUDE.md) for full rules):

```
GitHub Issue → Tests FIRST → Codex review tests → Implement →
Codex + Roast review → Fix → Verify tests → Commit → Close issue
```

## Paper Reference

- **TurboQuant**: arXiv 2504.19874 (ICLR 2026)
- **PolarQuant**: arXiv 2502.02617 (AISTATS 2026)
- **QJL**: arXiv 2406.03482

## Roadmap

| Phase | Status |
|-------|--------|
| Core algorithms (NumPy) | ✅ Complete — 141 tests, 100% coverage |
| Distortion validation | ✅ Complete — matches paper bounds |
| Outlier channel strategy | ✅ Complete — 2.5-bit/3.5-bit rates |
| Real model validation (Phase A) | ✅ Complete — rotation validated on Qwen3 |
| llama.cpp C port (Phase B) | ✅ **Working** — Metal GPU inference |
| Benchmarks | 🔄 In Progress |
| Metal shader optimization | ⏳ Next |
| MLX port | ⏳ Last |

## Target Hardware

- Apple M5 Max 128 GB (llama.cpp + Metal)
- RTX 3090 24 GB (llama.cpp + CUDA)

## Prerequisites

- **Python**: >= 3.10 (for prototype)
- **NumPy**: >= 1.24, **SciPy**: >= 1.10
- **cmake** + C/C++ compiler (for llama.cpp build)
- **Xcode Command Line Tools** (macOS Metal build)
- **Optional**: `torch`, `transformers`, `accelerate` (for real model validation — ~4GB download)

## Building from Source

### Python Prototype (validation & testing)

```bash
git clone https://github.com/TheTom/turboquant_plus.git
cd turboquant_plus
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Verify install (should see 141 passed, 0 failed)
python3 -m pytest tests/ -v --cov=turboquant --cov-fail-under=95
```

### llama.cpp Integration (production inference)

The llama.cpp port lives on a feature branch. To build:

```bash
# Prerequisites: cmake, C/C++ compiler, Xcode Command Line Tools (macOS)

# Clone our fork with TurboQuant support
git clone https://github.com/TheTom/llama-cpp-turboquant.git
cd llama-cpp-turboquant
# OR: if you already have llama.cpp, add our remote:
# git remote add turboquant https://github.com/TheTom/llama-cpp-turboquant.git
# git fetch turboquant feature/turboquant-kv-cache
# git checkout feature/turboquant-kv-cache

# Build with Metal (Apple Silicon) — requires macOS + Xcode
cmake -B build -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Build with CUDA (NVIDIA) — requires CUDA toolkit
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Verify turbo types appear
./build/bin/llama-server --help | grep turbo
# Expected: turbo3, turbo4 in cache-type-k options
```

> **Note**: The llama.cpp integration is a fork, not yet upstreamed. The C/Metal sources
> live in the fork repo, not in this Python prototype repo. See the
> [feature branch](https://github.com/TheTom/llama-cpp-turboquant/tree/feature/turboquant-kv-cache)
> for the full C implementation.

### Running with TurboQuant KV Cache

```bash
# Server mode (for Hermes Agent, Claude Code, etc.)
./build/bin/llama-server \
  -m models/Qwen3.5-27B-Q8_0.gguf \
  --alias "qwen-27b-turbo" \
  --jinja -ngl 99 -c 262144 -fa on \
  --cache-type-k turbo3 --cache-type-v turbo3 \
  -np 1 --metrics --host 0.0.0.0 --port 8080

# CLI mode (quick test)
./build/bin/llama-cli \
  -m models/your-model.gguf \
  -ngl 99 -c 2048 -fa on \
  --cache-type-k turbo3 --cache-type-v turbo3 \
  -n 100 -p "Hello world" --jinja
```

### Cache Type Options

| Flag | Bits/val | Compression vs fp16 | Description |
|------|----------|--------------------:|-------------|
| `turbo3` | 3.25 | **4.9×** | 2-bit PolarQuant + 1-bit QJL. Best compression. |
| `turbo4` | 4.25 | **3.8×** | 3-bit PolarQuant + 1-bit QJL. Better quality. |
| `q8_0` | 8 | 2.0× | llama.cpp default quantized cache. |
| `q4_0` | 4 | 4.0× | llama.cpp 4-bit cache. |

## License

Apache License 2.0. See [LICENSE](LICENSE).

Based on Google Research's TurboQuant paper (arXiv 2504.19874).
