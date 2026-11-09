@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Creating the Python environment...
    py -3.13 -m venv .venv
    if errorlevel 1 (
        python -c "import sys, struct; sys.exit(not (sys.version_info[:2] == (3, 13) and struct.calcsize('P') == 8))"
        if errorlevel 1 goto python_error
        python -m venv .venv
        if errorlevel 1 goto failed
    )
)
if not exist ".venv\compare-ready" (
    echo Installing Attentify's Python dependencies...
    ".venv\Scripts\python.exe" -m pip --version >nul 2>&1
    if errorlevel 1 (
        ".venv\Scripts\python.exe" -m ensurepip
        if errorlevel 1 goto failed
    )
    ".venv\Scripts\python.exe" -m pip install -r "..\face-tracking-prototype\requirements.txt"
    if errorlevel 1 goto failed
    echo ready>".venv\compare-ready"
)
".venv\Scripts\python.exe" compare.py %*
if errorlevel 1 goto failed
exit /b 0

:python_error
echo Install 64-bit Python 3.13 from python.org with the launcher or PATH option enabled.
:failed
echo The launcher stopped. See the error above.
pause
exit /b 1
