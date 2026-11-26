@echo off
setlocal
cd /d "%~dp0"
".venv-torch\Scripts\python.exe" -B scripts\play_headed.py %*
