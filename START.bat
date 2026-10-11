@echo off
setlocal EnableExtensions
title AI Image Upscaler - Universal CPU Portable
cd /d "%~dp0"

rem ---- Portable runtime paths: nothing needs to be installed ----
set "PYTHONHOME=%CD%\python"
set "PYTHONPATH=%CD%;%CD%\python\Lib\site-packages"
set "PATH=%CD%\python;%CD%\python\Lib\site-packages\numpy.libs;%CD%\python\Lib\site-packages\onnxruntime\capi;%PATH%"

if not exist "python\python.exe" (
  echo [ERROR] Portable runtime folder is missing.
  echo Please extract the ZIP completely before running START.bat.
  pause
  exit /b 1
)
if not exist "models\realesr-general-x4v3.onnx" (
  echo [ERROR] AI model files are missing.
  echo Please extract the ZIP completely before running START.bat.
  pause
  exit /b 1
)

rem ---- Verify Windows architecture ----
if /I "%PROCESSOR_ARCHITECTURE%"=="x86" if "%PROCESSOR_ARCHITEW6432%"=="" (
  echo [ERROR] This portable build requires 64-bit Windows.
  pause
  exit /b 1
)

rem ---- Preflight the AI runtime first, so failures are readable ----
>runtime_test.log 2>&1 python\python.exe -c "import os; os.environ['CUDA_VISIBLE_DEVICES']=''; import onnxruntime as ort; print('ONNX Runtime', ort.__version__); print('Providers:', ort.get_available_providers()); assert 'CPUExecutionProvider' in ort.get_available_providers()"
if errorlevel 1 (
  echo.
  echo ================================================
  echo AI runtime could not start on this PC.
  echo A diagnostic was saved as runtime_test.log
  echo ================================================
  echo.
  type runtime_test.log
  echo.
  echo This build does NOT require a separate Python install.
  echo If the message mentions MSVCP/VCRUNTIME, Windows C++ runtime is damaged or missing.
  echo.
  pause
  exit /b 1
)

echo Starting AI Image Upscaler - CPU Portable...
python\python.exe upscaler.py
if errorlevel 1 (
  echo.
  echo Program ended with an error.
  echo Check runtime_test.log and take a screenshot if needed.
  pause
)
endlocal
