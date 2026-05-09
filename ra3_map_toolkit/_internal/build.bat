@echo off
title Building RA3 Map Toolkit
chcp 65001 >nul

echo.
echo  =============================================
echo   BUILDING RA3 MAP TOOLKIT
echo  =============================================
echo.

cd /d "%~dp0"

REM ---- Check toolchain ----
where uv >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: uv is not installed.
    echo   Install: powershell -c "irm https://astral.sh/uv/install.ps1 ^| iex"
    pause
    exit /b 1
)

where bun >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: bun is not installed.
    echo   Install: powershell -c "irm https://bun.sh/install.ps1 ^| iex"
    pause
    exit /b 1
)

REM ---- Build the Python conversion engine ----
echo [1/2] Building ra3_engine.exe (Python engine)...
echo.

uv sync
if %ERRORLEVEL% neq 0 (
    echo Failed to sync Python dependencies!
    pause
    exit /b 1
)

uv pip install pyinstaller
if %ERRORLEVEL% neq 0 (
    echo Failed to install PyInstaller!
    pause
    exit /b 1
)

uv run python -m PyInstaller ^
    --onefile ^
    --name "ra3_engine" ^
    --add-data "templates;templates" ^
    --console ^
    batch_convert.py

if %ERRORLEVEL% neq 0 (
    echo Build of ra3_engine.exe FAILED!
    pause
    exit /b 1
)

if exist "dist\ra3_engine.exe" (
    move /Y "dist\ra3_engine.exe" "ra3_engine.exe" >nul
    if exist "build" rmdir /s /q "build"
    if exist "dist" rmdir /s /q "dist"
    if exist "ra3_engine.spec" del /q "ra3_engine.spec"
    echo   ra3_engine.exe ready.
) else (
    echo WARNING: ra3_engine.exe missing after build.
    pause
    exit /b 1
)

echo.
echo [2/2] Building ra3_map_toolkit.exe (termcn TUI)...
echo.

cd tui

if not exist node_modules (
    bun install
    if %ERRORLEVEL% neq 0 (
        echo Failed to install TUI dependencies!
        pause
        exit /b 1
    )
)

bun run build
if %ERRORLEVEL% neq 0 (
    echo Build of ra3_map_toolkit.exe FAILED!
    pause
    exit /b 1
)

cd ..

echo.
echo  =============================================
echo   BUILD SUCCESSFUL
echo  =============================================
echo.
echo Files in _internal\:
echo   - ra3_engine.exe       (Python engine)
echo   - ra3_map_toolkit.exe  (termcn TUI)
echo.
echo Distribute alongside "RA3 Map Toolkit.bat" at the parent folder.
echo.
pause
