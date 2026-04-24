@echo off
echo ============================================================
echo  PIPELINE MICRO - Ejecucion completa
echo ============================================================
echo.
echo  NOTA: Requiere haber ejecutado el pipeline macro antes
echo  (necesita trayectorias_proyectadas.parquet)
echo.

cd /d "%~dp0"

set /p ICAO="Codigo ICAO del aeropuerto a analizar: "

echo.
echo [1/3] Recorte y remuestreo del area terminal (%ICAO%)...
python ..\micro\recorte_micro.py %ICAO%
if %ERRORLEVEL% NEQ 0 (
    echo ERROR en recorte_micro.py
    pause
    exit /b 1
)
echo.

echo [2/3] Calculo de distancias ponderadas WED (%ICAO%)...
python ..\micro\distancias_micro.py %ICAO%
if %ERRORLEVEL% NEQ 0 (
    echo ERROR en distancias_micro.py
    pause
    exit /b 1
)
echo.

echo [3/3] Clustering HDBSCAN (%ICAO%)...
python ..\micro\clustering_micro.py %ICAO%
if %ERRORLEVEL% NEQ 0 (
    echo ERROR en clustering_micro.py
    pause
    exit /b 1
)
echo.

echo ============================================================
echo  PIPELINE MICRO COMPLETADO - %ICAO%
echo ============================================================
echo.
echo Para ver resultados ejecuta:
echo   python ..\dashboard\dashboard_micro.py
echo.
pause