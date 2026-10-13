@echo off
REM Crea el entorno virtual e instala las dependencias (solo la primera vez).
chcp 65001 >nul
cd /d "%~dp0"
set "PY="
py -3.12 --version >nul 2>&1 && set "PY=py -3.12"
if not defined PY py -3 --version >nul 2>&1 && set "PY=py -3"
if not defined PY python --version >nul 2>&1 && set "PY=python"
if not defined PY (
    echo [ERROR] No se encontro Python. Instala Python 3.12 desde https://www.python.org/downloads/
    echo         y marca la casilla "Add python.exe to PATH".
    pause
    exit /b 1
)
echo Usando: %PY%
if not exist ".venv\Scripts\python.exe" (
    %PY% -m venv .venv || goto :error
)
".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :error
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :error
echo.
echo Instalacion completa. Siguiente paso: ejecutar_todo_windows.bat o demo_windows.bat
pause
exit /b 0
:error
echo [ERROR] Fallo la instalacion. Revisa el mensaje de arriba.
pause
exit /b 1
