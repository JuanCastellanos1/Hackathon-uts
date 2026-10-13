@echo off
REM Regenera todo el pipeline (tabla, modelos, agente RL, presentacion) y corre las pruebas.
REM Uso: ejecutar_todo_windows.bat            (usa los datos crudos incluidos)
REM      ejecutar_todo_windows.bat --descargar (vuelve a descargar de SGC, DANE y UNGRD)
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Primero ejecuta instalar_windows.bat
    pause
    exit /b 1
)
".venv\Scripts\python.exe" run_all.py %* || goto :error
".venv\Scripts\python.exe" -m pytest -q || goto :error
echo.
echo Pipeline completo. Para la demo: demo_windows.bat
pause
exit /b 0
:error
echo [ERROR] Algo fallo. Revisa el mensaje de arriba.
pause
exit /b 1
