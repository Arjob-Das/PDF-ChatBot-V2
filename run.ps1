# =============================================================================
# PDF-Chatbot-V2 — Master Pipeline Launcher & Auto-Setup (PowerShell)
# =============================================================================

$ErrorActionPreference = "Stop"

# Set working directory to the script's root directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Enable UTF-8 encoding in console
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  PDF-Chatbot-V2 - Environment & Pipeline Setup" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

# -----------------------------------------------------------------------------
# 1. Directory Initialization
# -----------------------------------------------------------------------------
$RequiredDirs = @("data\pdfs", "logs", "vectorstore")
foreach ($dir in $RequiredDirs) {
    $targetPath = Join-Path $ScriptDir $dir
    if (-not (Test-Path -Path $targetPath)) {
        New-Item -ItemType Directory -Path $targetPath -Force | Out-Null
    }
}
Write-Host "[OK] Directories verified (data/pdfs, logs, vectorstore)." -ForegroundColor Green

# -----------------------------------------------------------------------------
# 2. Python Check & Installation
# -----------------------------------------------------------------------------
$SystemPython = $null
if (Get-Command python -ErrorAction SilentlyContinue) {
    $SystemPython = "python"
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $SystemPython = "py"
} elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    $SystemPython = "python3"
}

if (-not $SystemPython) {
    Write-Host "[!] Python not found in PATH. Attempting automated installation via winget..." -ForegroundColor Yellow
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        try {
            winget install Python.Python.3.11 --silent --accept-package-agreements --accept-source-agreements
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
            $SystemPython = "python"
        } catch {
            Write-Host "[ERROR] Winget installation failed. Please install Python 3.10+ manually." -ForegroundColor Red
            Exit 1
        }
    } else {
        Write-Host "[ERROR] Python is not installed. Please install Python 3.10+ from https://www.python.org/downloads/" -ForegroundColor Red
        Exit 1
    }
}

# -----------------------------------------------------------------------------
# 3. Virtual Environment Setup
# -----------------------------------------------------------------------------
$VenvDir = Join-Path $ScriptDir "venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$VenvPip = Join-Path $VenvDir "Scripts\pip.exe"

if (-not (Test-Path -Path $VenvPython)) {
    Write-Host "[*] Creating virtual environment at '$VenvDir'..." -ForegroundColor Yellow
    & $SystemPython -m venv "$VenvDir"
} else {
    Write-Host "[OK] Virtual environment detected at '$VenvDir'." -ForegroundColor Green
}

# -----------------------------------------------------------------------------
# 4. Python Package Dependencies Installation
# -----------------------------------------------------------------------------
Write-Host "[*] Verifying Python dependencies..." -ForegroundColor Gray
$CheckImportsCode = "import sys; pkgs = ['pymupdf', 'chromadb', 'sentence_transformers', 'rich', 'rank_bm25', 'requests', 'rapidocr_onnxruntime', 'torch', 'einops']; [__import__(p) for p in pkgs]"

$MissingPackages = $false
try {
    $proc = Start-Process -FilePath $VenvPython -ArgumentList "-c `"$CheckImportsCode`"" -NoNewWindow -Wait -PassThru -ErrorAction SilentlyContinue
    if ($proc.ExitCode -ne 0) {
        $MissingPackages = $true
    }
} catch {
    $MissingPackages = $true
}

if ($MissingPackages) {
    Write-Host "[*] Installing required packages from requirements.txt..." -ForegroundColor Yellow
    & $VenvPython -m pip install --upgrade pip | Out-Null
    
    # Check for NVIDIA GPU to install CUDA PyTorch
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        Write-Host "[*] NVIDIA GPU detected. Ensuring PyTorch with CUDA support..." -ForegroundColor Yellow
        & $VenvPython -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    }
    
    & $VenvPython -m pip install -r (Join-Path $ScriptDir "requirements.txt")
    Write-Host "[OK] Python dependencies installed successfully." -ForegroundColor Green
} else {
    Write-Host "[OK] Python dependencies already installed (skipped reinstall)." -ForegroundColor Green
}

# -----------------------------------------------------------------------------
# 5. Ollama Setup & Model Verification
# -----------------------------------------------------------------------------
$OllamaCmd = $null
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    $OllamaCmd = "ollama"
} else {
    $DefaultOllamaPath = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
    if (Test-Path -Path $DefaultOllamaPath) {
        $OllamaCmd = $DefaultOllamaPath
        $env:Path += ";$($env:LOCALAPPDATA)\Programs\Ollama"
    }
}

if (-not $OllamaCmd) {
    Write-Host "[!] Ollama not found. Attempting automated installation..." -ForegroundColor Yellow
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        try {
            winget install Ollama.Ollama --silent --accept-package-agreements --accept-source-agreements
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
            $OllamaCmd = "ollama"
        } catch {
            Write-Host "[!] Winget install failed. Downloading installer directly..." -ForegroundColor Yellow
            $InstallerPath = Join-Path $env:TEMP "OllamaSetup.exe"
            Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" -OutFile $InstallerPath
            Start-Process -FilePath $InstallerPath -Wait
            $OllamaCmd = "ollama"
        }
    } else {
        $InstallerPath = Join-Path $env:TEMP "OllamaSetup.exe"
        Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" -OutFile $InstallerPath
        Start-Process -FilePath $InstallerPath -Wait
        $OllamaCmd = "ollama"
    }
} else {
    Write-Host "[OK] Ollama installation detected." -ForegroundColor Green
}

# Check if Ollama daemon is running
$OllamaRunning = $false
try {
    $response = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 3 -ErrorAction SilentlyContinue
    if ($response) {
        $OllamaRunning = $true
    }
} catch {
    $OllamaRunning = $false
}

if (-not $OllamaRunning -and $OllamaCmd) {
    Write-Host "[*] Starting Ollama background service..." -ForegroundColor Yellow
    Start-Process -FilePath $OllamaCmd -ArgumentList "serve" -WindowStyle Hidden
    # Wait for Ollama service to start
    $Retries = 10
    while ($Retries -gt 0 -and -not $OllamaRunning) {
        Start-Sleep -Seconds 1
        try {
            $resp = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 2 -ErrorAction SilentlyContinue
            if ($resp) {
                $OllamaRunning = $true
                break
            }
        } catch {}
        $Retries--
    }
}

if ($OllamaRunning) {
    Write-Host "[OK] Ollama service is active." -ForegroundColor Green
    
    # Check if ANY models are installed in Ollama
    $ModelsOutput = ""
    try {
        $ModelsOutput = & $OllamaCmd list 2>$null | Out-String
    } catch {}

    $Lines = $ModelsOutput.Trim() -split "`r?`n"
    # Header is first line; check if any model rows exist
    if ($Lines.Count -gt 1) {
        $FirstModel = ($Lines[1] -split "\s+")[0]
        Write-Host "[OK] Detected installed Ollama model '$FirstModel' (skipping model download)." -ForegroundColor Green
    } else {
        # No models installed at all, pull default
        $TargetModel = "qwen3:8b"
        Write-Host "[*] No Ollama models found. Pulling '$TargetModel' (this may take a few minutes)..." -ForegroundColor Yellow
        & $OllamaCmd pull $TargetModel
        Write-Host "[OK] Model '$TargetModel' ready." -ForegroundColor Green
    }
} else {
    Write-Host "[!] Note: Ollama daemon not reachable on http://localhost:11434 (offline fallback mode)." -ForegroundColor Yellow
}

# -----------------------------------------------------------------------------
# 6. Execute Main Pipeline
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  Launching PDF-Chatbot-V2 Pipeline..." -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

$MainScript = Join-Path $ScriptDir "main.py"
& $VenvPython $MainScript @args
