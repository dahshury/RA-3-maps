# RA3 Map Processing Python Tools

This directory contains Python tools for processing RA3 maps for AI training.

## Structure

- `map_processor/` - Python package for map processing
  - `csharp_bridge.py` - Interface to C# MapCoreLib
  - `utils.py` - Utility functions
- `tests/` - pytest tests for the pipeline
  - `test_map_parsing.py` - Tests for map parsing
  - `test_map_reconstruction.py` - Tests for reconstruction pipeline
  - `test_pipeline.py` - Integration tests
- `scripts/` - Utility scripts
- `requirements.txt` - Python dependencies

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

### Optional: GPU acceleration (CuPy)

The visualization path supports an optional **GPU backend** via [CuPy](https://cupy.dev/). This speeds up the heavy pixel/array math, but image encoding (PNG) still happens on CPU.

- Install CuPy separately (recommended to pick a CUDA-matched wheel, e.g. `cupy-cuda12x`).
- Or install the project’s GPU extra (pulls `cupy-cuda12x`):

```bash
pip install -e .[gpu]
```

2. Build the C# CLI tool (see `scripts/create_csharp_cli_placeholder.md`)

## Running Tests

Run all tests:
```bash
pytest tests/
```

Run specific test file:
```bash
pytest tests/test_map_parsing.py
```

Run with verbose output:
```bash
pytest tests/ -v
```

Run a specific test:
```bash
pytest tests/test_map_reconstruction.py::test_reconstruct_map_from_json -v
```

## Test Structure

The tests are organized to verify:

1. **Parsing Tests** (`test_map_parsing.py`):
   - Finding map files
   - Parsing maps to JSON
   - Extracting metadata

2. **Reconstruction Tests** (`test_map_reconstruction.py`):
   - Full pipeline: parse → reconstruct → parse
   - Consistency checks
   - File size validation

3. **Pipeline Tests** (`test_pipeline.py`):
   - End-to-end integration tests
   - Consistency across multiple runs

## Current Status

⚠️ **Note**: The tests require a C# CLI tool to be built first. 
See `scripts/create_csharp_cli_placeholder.md` for details.

The tests will automatically skip if the C# tool is not found, so you can run them to see what's expected.

## Next Steps

1. Implement the C# CLI tool (or use Python.NET as alternative)
2. Run tests to verify the pipeline works
3. Extract training dataset from all maps
4. Prepare data for AI model training

## Visualization scripts

Generate images for a single map:

```bash
python scripts/generate_map_image.py "<path/to/map.map>" "<output/dir>" --gpu
```

Batch visualize a directory of maps (optionally in parallel):

```bash
python scripts/visualize_all_maps.py "../RA3 Official maps" "../test_output/all_maps_training" --workers 4 --gpu
```
