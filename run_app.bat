@echo off
title Econometric Diagnostics & Remedies Web Platform
echo ========================================================
echo   Launching Econometric Diagnostics & Remedies Platform
echo   Web Server: http://localhost:8000
echo ========================================================
echo.

start "" "http://localhost:8000"
python -m uvicorn app.server:app --host 127.0.0.1 --port 8000 --reload
pause
