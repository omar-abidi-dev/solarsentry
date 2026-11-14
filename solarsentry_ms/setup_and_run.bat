@echo off
REM SolarSentry MindSpore: one-click setup, conversion, test and training (Windows 10/11).
REM Put the solarsentry_ms folder inside your SolarSentry prototype folder, then double-click this file.
setlocal
cd /d "%~dp0"

echo === 1/5 Checking for Python 3.11 ===
py -3.11 --version >nul 2>&1
if errorlevel 1 (
  echo Python 3.11 not found. Install it from https://www.python.org/downloads/release/python-3119/
  echo In the installer, tick "Add python.exe to PATH", then run this file again.
  pause & exit /b 1
)

echo === 2/5 Creating environment and installing MindSpore (a few minutes) ===
if not exist .venv py -3.11 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip -q
pip install -r requirements.txt torch -q || (echo Install failed. & pause & exit /b 1)

echo === 3/5 Converting your trained PyTorch model ===
if not exist ..\model\model.pt (
  echo ..\model\model.pt not found. Put solarsentry_ms inside the SolarSentry folder.
  pause & exit /b 1
)
if not exist model mkdir model
python convert_pt_to_ms.py --pt ..\model\model.pt --out model || (pause & exit /b 1)
copy /y ..\model\normalization.json model\ >nul

echo === 4/5 Testing on the sample images (expect 18 confirmed faults, 2 to check) ===
python solarsentry_inspect.py ..\sample_images --out inspection_report.csv || (pause & exit /b 1)

echo === 5/5 Retraining in MindSpore ===
choice /c QFS /m "Q = quick 2-epoch test (5 min), F = full 30 epochs (about 75 min), S = skip"
if errorlevel 3 goto done
if errorlevel 2 (set EP=30) else (set EP=2)
if not exist data\InfraredSolarModules\module_metadata.json (
  echo Downloading the Raptor Maps dataset, 15 MB...
  if not exist data mkdir data
  curl -L -o data\ism.zip https://github.com/RaptorMaps/InfraredSolarModules/raw/master/2020-02-14_InfraredSolarModules.zip || (pause & exit /b 1)
  tar -xf data\ism.zip -C data
  del data\ism.zip
)
python train.py --data data\InfraredSolarModules --out out --epochs %EP% || (pause & exit /b 1)
echo Training results are in out\metrics.json. To use the new model:
echo   copy out\model.ckpt model\  and  copy out\normalization.json model\

:done
echo All done.
pause
