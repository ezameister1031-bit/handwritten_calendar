@echo off
cd /d "%~dp0"
set VENV=%USERPROFILE%\.venvs\handwritten_calendar
if not exist "%VENV%\Scripts\activate.bat" (
  echo Please run setup.bat first.
  pause
  exit /b 1
)
call "%VENV%\Scripts\activate.bat"
streamlit run app.py
pause
