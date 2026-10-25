@echo off
setlocal
cd /d "%~dp0"
title PolicyLens AI

rem ---- 1. Find Python ----
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (
  where python >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo Python was not found on this computer.
  echo Install Python 3.10 or 3.11 from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" in the installer, then double-click this file again.
  pause
  exit /b 1
)

rem ---- 2. First run only: create the environment and install packages ----
if not exist ".venv\Scripts\python.exe" (
  echo Setting up PolicyLens for the first time. This can take 5 to 10 minutes.
  %PY% -m venv .venv
  if errorlevel 1 goto :fail
)
if not exist ".venv\policylens_ready.txt" (
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 goto :fail
  ".venv\Scripts\python.exe" -m pip install -r requirements-ml.txt
  if errorlevel 1 goto :fail
  echo ready> ".venv\policylens_ready.txt"
)

rem ---- 3. Stop Streamlit asking for an email address on first launch ----
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
  mkdir "%USERPROFILE%\.streamlit" 2>nul
  echo [general]> "%USERPROFILE%\.streamlit\credentials.toml"
  echo email = "">> "%USERPROFILE%\.streamlit\credentials.toml"
)

rem ---- 4. Start the app (your browser opens by itself) ----
echo.
echo PolicyLens is starting. Your browser will open in a few seconds.
echo Keep this window open while you use the app. Close it to stop PolicyLens.
echo.
".venv\Scripts\python.exe" -m streamlit run app\streamlit_app.py
if errorlevel 1 goto :fail
exit /b 0

:fail
echo.
echo Something went wrong. Copy the messages above and send them to whoever helps you with this project.
pause
exit /b 1
