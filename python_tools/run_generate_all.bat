@echo off
setlocal enabledelayedexpansion

REM One-click batch visualization runner.
REM Defaults:
REM   input:  ..\RA3 Official maps
REM   output: test_output
REM
REM Usage:
REM   run_generate_all.bat
REM   run_generate_all.bat --workers 8
REM   run_generate_all.bat --no-training-config
REM

cd /d "%~dp0"

uv run python scripts\visualize_all_maps.py "..\RA3 Official maps" "test_output" %*

endlocal










