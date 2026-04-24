Copy

@echo off
echo Eliminando contenido de resultados/micro...
cd /d "%~dp0"
del /Q "..\..\resultados\micro\*.*" 2>nul
echo Limpieza completada.
pause