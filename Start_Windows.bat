@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1" %*
if errorlevel 1 (
  echo.
  echo Не удалось запустить программу. Скопируйте ошибку из этого окна.
  pause
)
