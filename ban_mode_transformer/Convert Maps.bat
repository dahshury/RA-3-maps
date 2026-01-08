@echo off
REM Batch script to convert maps to ban mode
REM Edit the paths below to match your setup

set SCRIPT_PATH=%~dp0_internal\transform_to_ban_mode.py
set BASE_MAP=..\RA3 Official maps\2 II\map_mp_2_rao1.map
set TEMPLATE=..\RA3 Official maps\BanMode 1 PLAYER MAPS\Ban_II_2.1\Ban_II_2.1.map
set OUTPUT=converted_maps\Ban_II_2.1_test\Ban_II_2.1_test.map

echo Converting map to ban mode...
python "%SCRIPT_PATH%" --in "%BASE_MAP%" --out "%OUTPUT%" --template "%TEMPLATE%"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Conversion successful!
    echo Output saved to: %OUTPUT%
) else (
    echo.
    echo Conversion failed with error code %ERRORLEVEL%
    pause
)
