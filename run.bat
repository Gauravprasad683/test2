@echo off
cd /d "%~dp0"
echo ========================================================
echo Starting Elite Sports Medicine Streamlit Application...
echo ========================================================
"C:\Program Files\Python315\python.exe" -m streamlit run app.py
if %errorlevel% neq 0 (
    echo.
    echo Trying fallback py launcher...
    py -m streamlit run app.py
)
pause
