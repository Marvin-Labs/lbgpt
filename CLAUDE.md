# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LBGPT (Load Balancing ChatGPT) is a Python library that provides an enhanced wrapper around OpenAI's API with load balancing, caching, and multi-provider support via LiteLLM.

## Essential Commands

### Development Setup
```bash
# Install dependencies
task install

# Generate requirements from .in files
task requirements-generate  # or: task rg

# Start Docker services (Qdrant, Redis, MinIO)
task docker:start

# Stop Docker services
task docker:stop
```

### Code Quality
```bash
# Run linting and formatting (Black + isort)
task lint

# Run tests with Docker services
task test  # or: task tests

# Run tests directly
pytest

# Run a specific test
pytest tests/test_chatgpt.py::test_basic_request
```

## Architecture Overview

### Core Components

1. **Base Class** (`src/lbgpt/base.py`):
   - `_BaseGPT`: Abstract base providing rate limiting, caching, and async semaphore management
   - All GPT implementations inherit from this

2. **Main Implementations** (`src/lbgpt/lbgpt.py`):
   - `ChatGPT`: Direct OpenAI API wrapper
   - `AzureGPT`: Azure OpenAI wrapper with deployment mapping
   - `LoadBalancedGPT`: Legacy load balancer (kept for compatibility)
   - `MultiLoadBalancedGPT`: Advanced load balancer supporting multiple endpoints
   - `LiteLlmRouter`: Multi-provider support via LiteLLM

3. **Caching System**:
   - **Standard Cache**: Exact match caching using any dict-like backend
   - **Semantic Cache**: Similarity-based caching using embeddings
   - Implementations in `src/lbgpt/caches/`:
     - FAISS (`faiss_cache.py`)
     - Qdrant (`qdrant_cache.py`)
     - S3 (`s3_cache.py`)

4. **Load Balancing** (`src/lbgpt/allocation.py`):
   - `random_allocation_function`: Weighted random selection
   - `max_headroom_allocation_function`: Selects based on rate limit headroom

### Key Design Patterns

- **Async-first**: All API calls use asyncio with configurable parallelism via semaphores
- **Two-tier caching**: Checks semantic cache first, then standard cache
- **Automatic retries**: Via tenacity with configurable attempts
- **Usage tracking**: Maintains in-memory usage cache for rate limit management

### Testing Approach

- Uses `pytest-recording` (VCR.py) to record/replay API interactions
- Async tests with `pytest-asyncio`
- Test fixtures in `conftest.py` provide mock clients and caches
- Docker services required for integration tests (Qdrant, Redis, MinIO)

## Important Notes

- **Python 3.11+** required
- **Async context managers**: All GPT classes must be used with `async with`
- **Model mapping**: AzureGPT requires `azure_model_map` to map OpenAI model names to Azure deployment IDs
- **Rate limiting**: Implemented via semaphores, configurable per instance
- **Usage tracking**: The `output_tokens` field in Usage objects represents total tokens (not completion tokens)