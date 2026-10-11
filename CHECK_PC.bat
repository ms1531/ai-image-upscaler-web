@echo off
setlocal
cd /d "%~dp0"
set "PYTHONHOME=%CD%\python"
set "PYTHONPATH=%CD%;%CD%\python\Lib\site-packages"
set "PATH=%CD%\python;%CD%\python\Lib\site-packages\numpy.libs;%CD%\python\Lib\site-packages\onnxruntime\capi;%PATH%"
echo Windows: & ver
echo Architecture: %PROCESSOR_ARCHITECTURE% %PROCESSOR_ARCHITEW6432%
echo.
python\python.exe -c "import platform,sys; print('Portable Python:',sys.version); print('Machine:',platform.machine()); import numpy; print('NumPy:',numpy.__version__); import onnxruntime as ort; print('ONNX Runtime:',ort.__version__); print('Providers:',ort.get_available_providers())"
echo.
pause
