@echo off
REM Seestar S50 Pipeline - Setup Validation Script
REM Checks that all required tools are installed and accessible

setlocal enabledelayedexpansion
set ERRORS=0

echo ============================================
echo  Seestar S50 Pipeline - Setup Validation
echo ============================================
echo.

REM Check Python
echo [CHECK] Python 3.12+...
python --version 2>nul
if %errorlevel% neq 0 (
    echo [FAIL] Python not found in PATH
    set /a ERRORS+=1
) else (
    echo [OK]   Python found
)

REM Check Python dependencies
echo.
echo [CHECK] Python dependencies...

python -c "import astropy; print(f'  astropy {astropy.__version__}')" 2>nul
if %errorlevel% neq 0 (
    echo [FAIL] astropy not installed ^(pip install astropy^)
    set /a ERRORS+=1
)

python -c "import astroquery; print(f'  astroquery {astroquery.__version__}')" 2>nul
if %errorlevel% neq 0 (
    echo [FAIL] astroquery not installed ^(pip install astroquery^)
    set /a ERRORS+=1
)

python -c "import yaml; print('  PyYAML OK')" 2>nul
if %errorlevel% neq 0 (
    echo [FAIL] PyYAML not installed ^(pip install pyyaml^)
    set /a ERRORS+=1
)

python -c "import PIL; print(f'  Pillow {PIL.__version__}')" 2>nul
if %errorlevel% neq 0 (
    echo [FAIL] Pillow not installed ^(pip install Pillow^)
    set /a ERRORS+=1
)

python -c "import numpy; print(f'  numpy {numpy.__version__}')" 2>nul
if %errorlevel% neq 0 (
    echo [FAIL] numpy not installed ^(pip install numpy^)
    set /a ERRORS+=1
)

REM Check Siril CLI
echo.
echo [CHECK] Siril CLI...
siril-cli --version 2>nul
if %errorlevel% neq 0 (
    echo [FAIL] siril-cli not found in PATH
    echo        Install from https://siril.org/download/
    echo        Add to PATH: C:\Program Files\Siril\bin\
    set /a ERRORS+=1
) else (
    echo [OK]   siril-cli found
)

REM Check GraXpert
echo.
echo [CHECK] GraXpert...
graxpert --version 2>nul
if %errorlevel% neq 0 (
    where GraXpert-win64.exe >nul 2>nul
    if !errorlevel! neq 0 (
        echo [FAIL] GraXpert not found
        echo        Install via: pip install graxpert[cuda]
        echo        Or download standalone from: https://github.com/Steffenhir/GraXpert/releases
        set /a ERRORS+=1
    ) else (
        echo [OK]   GraXpert standalone found
    )
) else (
    echo [OK]   GraXpert ^(pip^) found
)

REM Check Cosmic Clarity
echo.
echo [CHECK] Cosmic Clarity...
where SetiAstroCosmicClarity.exe >nul 2>nul
if %errorlevel% neq 0 (
    echo [WARN] SetiAstroCosmicClarity.exe not found in PATH
    echo        Download from https://www.setiastro.com/cosmic-clarity
    echo        Add its directory to PATH or update tool path in pipeline.yaml
    echo        ^(Deconvolution can be skipped with --skip-deconvolution^)
) else (
    echo [OK]   Cosmic Clarity found
)

REM Check NVIDIA GPU / CUDA
echo.
echo [CHECK] NVIDIA GPU / CUDA...
nvidia-smi >nul 2>nul
if %errorlevel% neq 0 (
    echo [WARN] nvidia-smi not found - GPU acceleration may not be available
    echo        Install NVIDIA drivers from https://www.nvidia.com/drivers/
) else (
    echo [OK]   NVIDIA GPU detected
)

REM Check pipeline configuration files
echo.
echo [CHECK] Pipeline configuration files...

if exist pipeline.yaml (
    echo [OK]   pipeline.yaml found
    python -c "import yaml; yaml.safe_load(open('pipeline.yaml')); print('        Config valid')" 2>nul
    if !errorlevel! neq 0 (
        echo [FAIL] pipeline.yaml is not valid YAML
        set /a ERRORS+=1
    )
) else (
    echo [FAIL] pipeline.yaml not found in current directory
    set /a ERRORS+=1
)

if exist profiles.yaml (
    echo [OK]   profiles.yaml found
    python -c "import yaml; yaml.safe_load(open('profiles.yaml')); print('        Profiles valid')" 2>nul
    if !errorlevel! neq 0 (
        echo [FAIL] profiles.yaml is not valid YAML
        set /a ERRORS+=1
    )
) else (
    echo [FAIL] profiles.yaml not found in current directory
    set /a ERRORS+=1
)

REM Check Siril scripts
echo.
echo [CHECK] Siril scripts...
set SCRIPTS_OK=1
for %%F in (seestar_stack.ssf seestar_calibrate.ssf seestar_stretch.ssf seestar_platesolve.ssf) do (
    if exist %%F (
        echo [OK]   %%F
    ) else (
        echo [FAIL] %%F not found
        set /a ERRORS+=1
        set SCRIPTS_OK=0
    )
)

REM Summary
echo.
echo ============================================
if %ERRORS% equ 0 (
    echo  All checks passed! Pipeline is ready.
    echo.
    echo  Usage: python astro_pipeline.py "D:\Astro\M42_subs"
    echo ============================================
    exit /b 0
) else (
    echo  %ERRORS% check^(s^) failed. Fix the issues above.
    echo ============================================
    exit /b 1
)
