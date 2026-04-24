@echo off
echo ============================================================
echo  PIPELINE MACRO - Ejecucion completa
echo ============================================================
echo.

cd /d "%~dp0"

echo [1/6] Fusionando parquets...
python ..\preparacion\fusionar.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR en fusionar.py
    pause
    exit /b 1
)
echo.

echo [2/6] Limpieza de trayectorias...
python ..\preparacion\limpieza.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR en limpieza.py
    pause
    exit /b 1
)
echo.

echo [3/6] Proyeccion LCC...
python ..\preparacion\proyeccion.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR en proyeccion.py
    pause
    exit /b 1
)
echo.

echo [4/6] Remuestreo espacial...
python ..\preparacion\remuestreo_espacial.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR en remuestreo_espacial.py
    pause
    exit /b 1
)
echo.

echo [5/6] Matriz de distancias...
python ..\macro\distancias_macro.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR en distancias_macro.py
    pause
    exit /b 1
)
echo.

echo [6/6] Clustering HDBSCAN...
python ..\macro\clustering_macro.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR en clustering_macro.py
    pause
    exit /b 1
)
echo.

echo ============================================================
echo  PIPELINE COMPLETADO
echo ============================================================
echo.
echo Para ver resultados ejecuta:
echo   python ..\macro\caracterizacion_macro.py
echo   python ..\macro\visualizar_macro.py
echo   python ..\dashboard\dashboard.py
echo.
pause