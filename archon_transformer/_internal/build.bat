@echo off
title Building RA3 Archon Converter
color 0E

echo.
echo  =============================================
echo   BUILDING STANDALONE EXECUTABLE
echo  =============================================
echo.

cd /d "%~dp0"

REM Check for uv
where uv >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: uv is not installed.
    echo.
    echo Install it with:
    echo   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    echo.
    echo Or visit: https://docs.astral.sh/uv/
    pause
    exit /b 1
)

echo Installing dependencies...
uv sync
if %ERRORLEVEL% neq 0 (
    echo Failed to sync dependencies!
    pause
    exit /b 1
)

echo.
echo Installing PyInstaller...
uv pip install pyinstaller
if %ERRORLEVEL% neq 0 (
    echo Failed to install PyInstaller!
    pause
    exit /b 1
)

echo.
echo Building executable...
echo This may take a minute...
echo.

uv run pyinstaller ^
    --onefile ^
    --name "archon_converter" ^
    --add-data "templates;templates" ^
    --console ^
    batch_convert.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo Build FAILED!
    pause
    exit /b 1
)

echo.
echo  =============================================
echo   BUILD SUCCESSFUL!
echo  =============================================
echo.

REM Move exe to this folder
if exist "dist\archon_converter.exe" (
    move /Y "dist\archon_converter.exe" "archon_converter.exe" >nul
    echo Executable updated: _internal\archon_converter.exe
    
    REM Cleanup build artifacts
    if exist "build" rmdir /s /q "build"
    if exist "dist" rmdir /s /q "dist"
    if exist "archon_converter.spec" del /q "archon_converter.spec"
    
    echo.
    echo The exe is now ready. Users just need:
    echo  - Convert Maps.bat (in parent folder)
    echo  - _internal\ folder (this folder)
    echo  - maps_to_convert\ folder
    echo  - converted_maps\ folder
    echo.
) else (
    echo.
    echo WARNING: Could not find built executable!
    echo Check the dist\ folder manually.
)

echo.
pause

