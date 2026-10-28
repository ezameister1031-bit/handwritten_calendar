@echo off
cd /d "%~dp0"
set VENV=%USERPROFILE%\.venvs\handwritten_calendar
if not exist "%VENV%\Scripts\python.exe" (
  py -3 -m venv "%VENV%" 2>nul || python -m venv "%VENV%"
)
call "%VENV%\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install -r requirements.txt
if not exist ".streamlit\secrets.toml" copy ".streamlit\secrets.toml.example" ".streamlit\secrets.toml" >nul
if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
  (echo [general]& echo email = "")> "%USERPROFILE%\.streamlit\credentials.toml"
)
echo.
echo Setup finished.
echo Next: write your Gemini API key in .streamlit\secrets.toml, then double-click run.bat
pause
