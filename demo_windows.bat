@echo off
REM Abre la demo interactiva en el navegador (http://localhost:8501). Ctrl+C para cerrar.
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Primero ejecuta instalar_windows.bat
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app\app.py
pause
