@echo off
title CITIZEN Face Gate
cd /d %~dp0
echo Starting face recognition gate module...
py -3.11 face_gate.py
echo.
echo Face gate stopped.
pause
