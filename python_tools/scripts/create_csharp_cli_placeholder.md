# C# CLI Tool Setup

The Python tests expect a C# CLI tool that can parse and reconstruct maps.

## Required C# Tool Interface

The tool should support these commands:

```bash
# Parse map to JSON
MapProcessor.exe parse --input <map_path> --output <json_path>

# Reconstruct map from JSON
MapProcessor.exe reconstruct --input <json_path> --output <map_path>
```

## Implementation Options

### Option 1: Create Simple C# Console App

Create a new C# console project in `Ra3Solution/MapProcessor/` that:

1. Uses `MapCoreLib` to parse maps
2. Exports to JSON using Newtonsoft.Json
3. Reconstructs maps from JSON
4. Provides CLI interface

### Option 2: Use Existing Test Projects

Modify `Ra3SolutionTest` or create wrapper scripts that can be called from Python.

### Option 3: Temporary Workaround

For now, the Python tests will skip if the C# tool is not found. You can:

1. Implement the C# tool later
2. Or modify the Python code to use a different interface (e.g., Python.NET)

## Quick Start Implementation

See `AI_TRAINING_GUIDE.md` for examples of how to use MapCoreLib to parse and reconstruct maps.
The C# CLI tool should wrap these operations.











