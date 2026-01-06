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
