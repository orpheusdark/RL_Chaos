@echo off
REM Auto-push hook for Windows - calls PowerShell script
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0auto_push.ps1"
