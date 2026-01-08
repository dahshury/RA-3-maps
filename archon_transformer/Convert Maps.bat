@echo off
title RA3 Archon Map Converter
color 0A

echo.
echo  =============================================
echo   RA3 ARCHON MAP CONVERTER
echo  =============================================
echo.
echo  Place your .map files in the "maps_to_convert" folder
echo  Converted maps will appear in "converted_maps"
echo.
echo  Press any key to start conversion...
pause >nul
echo.

"%~dp0_internal\archon_converter.exe"

echo.
echo  Press any key to exit...
pause >nul
