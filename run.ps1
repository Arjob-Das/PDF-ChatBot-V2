# =============================================================================
# PDF-Chatbot-V2 — Master Pipeline: Setup → Ollama → Train → Docker → Launch
# =============================================================================
# End-to-end orchestrator that handles the complete lifecycle:
#   Step 1: Directory initialization
#   Step 2: Python check & installation
#   Step 3: Virtual environment & dependencies
#   Step 4: Ollama installation, daemon start & model pull
#   Step 5: Document ingestion / training (local)
#   Step 6: Docker image build
#   Step 7: Start Docker container & open browser
#
# Usage:
#   .\run.ps1                  # Full pipeline (default: train → Docker → browser)
#   .\run.ps1 -cli             # Train → launch interactive CLI chatbot (no Docker)
#   .\run.ps1 --skip-ingest    # Skip training, use existing index
#   .\run.ps1 --reset          # Wipe index and re-train from scratch
#   .\run.ps1 --no-docker      # Skip Docker, run local web server instead
#   .\run.ps1 --local          # Same as -cli
#   .\run.ps1 --no-browser     # Don't auto-open browser at the end
# =============================================================================

$SkipIngest = $false
$Reset = $false
$NoDocker = $false
$Local = $false
$NoBrowser = $false
$ForceCPU = $false
$PdfDir = ""
$BatchSize = 0

for ($i = 0; $i -lt $args.Count; $i++) {
    switch -Regex ($args[$i]) {
        '^-{1,2}skip-ingest$' { $SkipIngest = $true }
        '^-{1,2}reset$'       { $Reset = $true }
        '^-{1,2}no-docker$'   { $NoDocker = $true }
        '^-{1,2}local$'       { $Local = $true }
        '^-{1,2}cli$'         { $Local = $true }
        '^-{1,2}no-browser$'  { $NoBrowser = $true }
        '^-{1,2}cpu$'         { $ForceCPU = $true }
        '^-{1,2}pdf-dir$'     { if ($i + 1 -lt $args.Count) { $i++; $PdfDir = $args[$i] } }
        '^-{1,2}batch-size$'  { if ($i + 1 -lt $args.Count) { $i++; $BatchSize = [int]$args[$i] } }
    }
}

$ErrorActionPreference = "Stop"

# Set working directory to the script's root directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Enable UTF-8 encoding in console
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  PDF-Chatbot-V2 — Full Pipeline Launcher" -ForegroundColor Cyan
Write-Host "  Setup → Ollama → Train → Docker → Launch" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Determine total steps based on mode
$TotalSteps = 7
if ($Local -or $NoDocker) { $TotalSteps = 6 }

# =============================================================================
# Step 1: Directory Initialization
# =============================================================================
Write-Host "[Step 1/$TotalSteps] Initializing directories..." -ForegroundColor White
$RequiredDirs = @("data\pdfs", "logs", "vectorstore")
foreach ($dir in $RequiredDirs) {
    $targetPath = Join-Path $ScriptDir $dir
    if (-not (Test-Path -Path $targetPath)) {
        New-Item -ItemType Directory -Path $targetPath -Force | Out-Null
    }
}
Write-Host "  [OK] Directories verified (data/pdfs, logs, vectorstore)." -ForegroundColor Green

# =============================================================================
# Step 2: Python Check & Installation
# =============================================================================
Write-Host "[Step 2/$TotalSteps] Checking Python..." -ForegroundColor White
$SystemPython = $null
if (Get-Command python -ErrorAction SilentlyContinue) {
    $SystemPython = "python"
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $SystemPython = "py"
} elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    $SystemPython = "python3"
}

if (-not $SystemPython) {
    Write-Host "  [!] Python not found. Attempting install via winget..." -ForegroundColor Yellow
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        try {
            winget install Python.Python.3.11 --silent --accept-package-agreements --accept-source-agreements
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
            $SystemPython = "python"
        } catch {
            Write-Host "  [ERROR] Install failed. Get Python 3.10+ from https://www.python.org/downloads/" -ForegroundColor Red
            Exit 1
        }
    } else {
        Write-Host "  [ERROR] Python not installed. Get Python 3.10+ from https://www.python.org/downloads/" -ForegroundColor Red
        Exit 1
    }
}
Write-Host "  [OK] Python found: $SystemPython" -ForegroundColor Green

# =============================================================================
# Step 3: Virtual Environment & Dependencies
# =============================================================================
Write-Host "[Step 3/$TotalSteps] Setting up virtual environment & dependencies..." -ForegroundColor White
$VenvDir = Join-Path $ScriptDir "venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$VenvPip = Join-Path $VenvDir "Scripts\pip.exe"

if (-not (Test-Path -Path $VenvPython)) {
    Write-Host "  [*] Creating virtual environment..." -ForegroundColor Yellow
    & $SystemPython -m venv "$VenvDir"
    Write-Host "  [OK] Virtual environment created." -ForegroundColor Green
} else {
    Write-Host "  [OK] Virtual environment exists." -ForegroundColor Green
}

# Check core dependencies
$CheckImportsCode = "import sys; pkgs = ['pymupdf', 'chromadb', 'sentence_transformers', 'rich', 'rank_bm25', 'requests', 'rapidocr_onnxruntime', 'torch', 'einops']; [__import__(p) for p in pkgs]"
$MissingCore = $false
try {
    $proc = Start-Process -FilePath $VenvPython -ArgumentList "-c `"$CheckImportsCode`"" -NoNewWindow -Wait -PassThru -ErrorAction SilentlyContinue
    if ($proc.ExitCode -ne 0) { $MissingCore = $true }
} catch { $MissingCore = $true }

if ($MissingCore) {
    Write-Host "  [*] Installing core dependencies..." -ForegroundColor Yellow
    & $VenvPython -m pip install --upgrade pip 2>&1 | Out-Null

    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        Write-Host "  [*] NVIDIA GPU detected — installing CUDA PyTorch..." -ForegroundColor Yellow
        & $VenvPython -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    }

    & $VenvPython -m pip install -r (Join-Path $ScriptDir "requirements.txt")
    Write-Host "  [OK] Core dependencies installed." -ForegroundColor Green
} else {
    Write-Host "  [OK] Core dependencies present." -ForegroundColor Green
}

# Check web dependencies
$CheckWebImports = "import fastapi; import uvicorn; import sse_starlette"
$MissingWeb = $false
try {
    $proc = Start-Process -FilePath $VenvPython -ArgumentList "-c `"$CheckWebImports`"" -NoNewWindow -Wait -PassThru -ErrorAction SilentlyContinue
    if ($proc.ExitCode -ne 0) { $MissingWeb = $true }
} catch { $MissingWeb = $true }

if ($MissingWeb) {
    Write-Host "  [*] Installing web dependencies..." -ForegroundColor Yellow
    & $VenvPython -m pip install -r (Join-Path $ScriptDir "web\requirements.txt") --quiet
    Write-Host "  [OK] Web dependencies installed." -ForegroundColor Green
}

# =============================================================================
# Step 4: Ollama Installation, Daemon Start & Model Pull
# =============================================================================
Write-Host "[Step 4/$TotalSteps] Setting up Ollama..." -ForegroundColor White

# 4a. Find or install Ollama
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
    Write-Host "  [!] Ollama not found. Installing..." -ForegroundColor Yellow
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        try {
            winget install Ollama.Ollama --silent --accept-package-agreements --accept-source-agreements
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
            $OllamaCmd = "ollama"
        } catch {
            Write-Host "  [!] Winget failed. Downloading installer..." -ForegroundColor Yellow
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
    Write-Host "  [OK] Ollama found." -ForegroundColor Green
}

# 4b. Start Ollama daemon if not running
$OllamaRunning = $false
try {
    $response = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 3 -ErrorAction SilentlyContinue
    if ($response) { $OllamaRunning = $true }
} catch {}

if (-not $OllamaRunning -and $OllamaCmd) {
    Write-Host "  [*] Starting Ollama daemon..." -ForegroundColor Yellow
    Start-Process -FilePath $OllamaCmd -ArgumentList "serve" -WindowStyle Hidden
    $Retries = 15
    while ($Retries -gt 0 -and -not $OllamaRunning) {
        Start-Sleep -Seconds 1
        try {
            $resp = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 2 -ErrorAction SilentlyContinue
            if ($resp) { $OllamaRunning = $true; break }
        } catch {}
        $Retries--
    }
}

if ($OllamaRunning) {
    Write-Host "  [OK] Ollama daemon is running." -ForegroundColor Green
} else {
    Write-Host "  [WARNING] Ollama daemon not reachable. Docker will handle Ollama internally." -ForegroundColor Yellow
}

# 4c. Pull model if needed (and Ollama is running)
if ($OllamaRunning -and $OllamaCmd) {
    $ModelsOutput = ""
    try { $ModelsOutput = & $OllamaCmd list 2>$null | Out-String } catch {}

    $Lines = $ModelsOutput.Trim() -split "`r?`n"
    if ($Lines.Count -gt 1) {
        $FirstModel = ($Lines[1] -split "\s+")[0]
        Write-Host "  [OK] Model available: '$FirstModel'" -ForegroundColor Green
    } else {
        $TargetModel = "qwen3:8b"
        Write-Host "  [*] No models found. Pulling '$TargetModel' (may take several minutes)..." -ForegroundColor Yellow
        & $OllamaCmd pull $TargetModel
        Write-Host "  [OK] Model '$TargetModel' pulled successfully." -ForegroundColor Green
    }
}

# =============================================================================
# Step 5: Document Ingestion / Training (Always Local)
# =============================================================================
Write-Host "[Step 5/$TotalSteps] Document ingestion & training..." -ForegroundColor White

$RunIngestion = $true
if ($SkipIngest) {
    $RunIngestion = $false
    Write-Host "  [*] Skipping ingestion (--skip-ingest)." -ForegroundColor Gray
}

# Check if vectorstore already has data
if ($RunIngestion -and -not $Reset) {
    $VectorstorePath = Join-Path $ScriptDir "vectorstore"
    $Bm25Path = Join-Path $VectorstorePath "bm25_index.pkl"

    if ((Test-Path $VectorstorePath) -and (Test-Path $Bm25Path)) {
        $CheckScript = @"
import sys; sys.path.insert(0, r'$ScriptDir')
import chromadb
client = chromadb.PersistentClient(path=r'$VectorstorePath')
try:
    col = client.get_collection('pdf_chatbot_v2'); count = col.count()
    print(f'EXISTS:{count}') if count > 0 else print('EMPTY')
except: print('MISSING')
"@
        $IndexStatus = & $VenvPython -c $CheckScript 2>$null

        if ($IndexStatus -and $IndexStatus.ToString().StartsWith("EXISTS:")) {
            $ChunkCount = $IndexStatus.ToString().Split(":")[1]
            Write-Host "  [OK] Existing index found ($ChunkCount chunks). Skipping training." -ForegroundColor Green
            Write-Host "       Use --reset to force full re-ingestion." -ForegroundColor Gray
            $RunIngestion = $false
        }
    }
}

if ($RunIngestion) {
    $PdfDirectory = if ($PdfDir) { $PdfDir } else { Join-Path $ScriptDir "data\pdfs" }
    $PdfCount = (Get-ChildItem -Path $PdfDirectory -Filter "*.pdf" -Recurse -ErrorAction SilentlyContinue | Measure-Object).Count

    if ($PdfCount -eq 0) {
        Write-Host "  [!] No PDF files found in '$PdfDirectory'." -ForegroundColor Yellow
        if (-not (Test-Path (Join-Path $ScriptDir "vectorstore\bm25_index.pkl"))) {
            Write-Host "  [ERROR] No PDFs and no existing index. Add PDFs to 'data/pdfs/' and re-run." -ForegroundColor Red
            Exit 1
        }
        Write-Host "  [*] Using existing vectorstore index." -ForegroundColor Gray
    } else {
        Write-Host "  [*] Found $PdfCount PDF(s). Starting ingestion pipeline..." -ForegroundColor Cyan

        $IngestArgs = @("--ocr")
        if ($Reset) { $IngestArgs += "--reset" }
        if ($PdfDir) { $IngestArgs += "--pdf-dir"; $IngestArgs += $PdfDir }
        if ($BatchSize -gt 0) { $IngestArgs += "--batch-size"; $IngestArgs += $BatchSize }

        & $VenvPython (Join-Path $ScriptDir "ingest.py") @IngestArgs

        if ($LASTEXITCODE -ne 0) {
            Write-Host "  [ERROR] Ingestion failed. Check logs/ for details." -ForegroundColor Red
            Exit 1
        }
        Write-Host "  [OK] Training complete!" -ForegroundColor Green
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# MODE: --local → Original interactive CLI chat (no web, no Docker)
# ─────────────────────────────────────────────────────────────────────────────
if ($Local) {
    Write-Host "[Step 6/$TotalSteps] Launching interactive CLI chat..." -ForegroundColor White
    Write-Host ""
    & $VenvPython (Join-Path $ScriptDir "main.py") --skip-ingest
    Exit 0
}

# ─────────────────────────────────────────────────────────────────────────────
# MODE: --no-docker → Local FastAPI web server (requires local Ollama)
# ─────────────────────────────────────────────────────────────────────────────
if ($NoDocker) {
    Write-Host "[Step 6/$TotalSteps] Launching local web server (no Docker)..." -ForegroundColor White

    if (-not $OllamaRunning) {
        Write-Host "  [ERROR] Ollama is not running. Start Ollama first or use Docker mode." -ForegroundColor Red
        Exit 1
    }

    Write-Host ""
    Write-Host "  ┌─────────────────────────────────────────────┐" -ForegroundColor Green
    Write-Host "  │  Web UI:  http://localhost:8000              │" -ForegroundColor Green
    Write-Host "  │  Press Ctrl+C to stop                       │" -ForegroundColor Green
    Write-Host "  └─────────────────────────────────────────────┘" -ForegroundColor Green
    Write-Host ""

    if (-not $NoBrowser) { Start-Process "http://localhost:8000" }
    & $VenvPython -m uvicorn web.app:app --host 0.0.0.0 --port 8000 --reload
    Exit 0
}

# =============================================================================
# Step 6: Docker Build
# =============================================================================
Write-Host "[Step 6/$TotalSteps] Building Docker serving image..." -ForegroundColor White

# 6a. Check Docker availability
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "  [!] Docker not found. Install Docker Desktop:" -ForegroundColor Yellow
    Write-Host "      https://www.docker.com/products/docker-desktop/" -ForegroundColor Gray
    Write-Host "  [*] Falling back to local web server mode..." -ForegroundColor Yellow
    Write-Host ""

    if (-not $OllamaRunning) {
        Write-Host "  [ERROR] Neither Docker nor Ollama available. Cannot serve." -ForegroundColor Red
        Exit 1
    }

    Write-Host "  ┌─────────────────────────────────────────────┐" -ForegroundColor Green
    Write-Host "  │  Web UI:  http://localhost:8000              │" -ForegroundColor Green
    Write-Host "  │  Press Ctrl+C to stop                       │" -ForegroundColor Green
    Write-Host "  └─────────────────────────────────────────────┘" -ForegroundColor Green
    Write-Host ""
    if (-not $NoBrowser) { Start-Process "http://localhost:8000" }
    & $VenvPython -m uvicorn web.app:app --host 0.0.0.0 --port 8000 --reload
    Exit 0
}

# 6b. Check Docker daemon
$DockerRunning = $false
try {
    docker info 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { $DockerRunning = $true }
} catch {}

if (-not $DockerRunning) {
    Write-Host "  [!] Docker daemon not running. Start Docker Desktop first." -ForegroundColor Yellow
    Write-Host "  [*] Falling back to local web server mode..." -ForegroundColor Yellow

    if (-not $OllamaRunning) {
        Write-Host "  [ERROR] Neither Docker daemon nor Ollama running." -ForegroundColor Red
        Exit 1
    }

    Write-Host ""
    Write-Host "  ┌─────────────────────────────────────────────┐" -ForegroundColor Green
    Write-Host "  │  Web UI:  http://localhost:8000              │" -ForegroundColor Green
    Write-Host "  │  Press Ctrl+C to stop                       │" -ForegroundColor Green
    Write-Host "  └─────────────────────────────────────────────┘" -ForegroundColor Green
    Write-Host ""
    if (-not $NoBrowser) { Start-Process "http://localhost:8000" }
    & $VenvPython -m uvicorn web.app:app --host 0.0.0.0 --port 8000 --reload
    Exit 0
}

Write-Host "  [OK] Docker is available." -ForegroundColor Green

# 6c. Verify vectorstore exists
if (-not (Test-Path (Join-Path $ScriptDir "vectorstore\bm25_index.pkl"))) {
    Write-Host "  [ERROR] No trained index found. Run without --skip-ingest first." -ForegroundColor Red
    Exit 1
}

# 6d. Build image
Write-Host "  [*] Building 'pdf-chatbot-v2:latest'..." -ForegroundColor Yellow
$env:DOCKER_BUILDKIT=1; docker build -f Dockerfile.serve -t pdf-chatbot-v2:latest .

if ($LASTEXITCODE -ne 0) {
    Write-Host "  [ERROR] Docker build failed. Check Dockerfile.serve." -ForegroundColor Red
    Exit 1
}
Write-Host "  [OK] Docker image built." -ForegroundColor Green

# =============================================================================
# Step 7: Start Docker Container & Open Browser
# =============================================================================
Write-Host "[Step 7/$TotalSteps] Starting Docker container..." -ForegroundColor White

# 7a. Stop & remove any existing container
$ExistingContainer = docker ps -aq --filter "name=pdf-chatbot-v2" 2>$null
if ($ExistingContainer) {
    Write-Host "  [*] Removing existing container..." -ForegroundColor Yellow
    docker stop pdf-chatbot-v2 2>$null | Out-Null
    docker rm pdf-chatbot-v2 2>$null | Out-Null
}

# 7b. Detect GPU
$UseGPU = $false
if ($ForceCPU) {
    Write-Host "  [*] CPU mode requested via --cpu flag." -ForegroundColor Yellow
} elseif (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    try {
        nvidia-smi 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { $UseGPU = $true }
    } catch {}
}

# 7c. Stop local Ollama (Docker runs its own Ollama — avoid port conflict on 11434)
if ($OllamaRunning) {
    Write-Host "  [*] Stopping local Ollama (Docker container runs its own)..." -ForegroundColor Yellow
    try {
        Get-Process -Name "ollama" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    } catch {}
    Start-Sleep -Seconds 2
}

# 7d. Run container
$VectorstoreFull = (Resolve-Path (Join-Path $ScriptDir "vectorstore")).Path
$OllamaHostDir = Join-Path $env:USERPROFILE ".ollama"
if (-not (Test-Path $OllamaHostDir)) { New-Item -ItemType Directory -Path $OllamaHostDir -Force | Out-Null }
$HfHostDir = Join-Path $env:USERPROFILE ".cache\huggingface"
if (-not (Test-Path $HfHostDir)) { New-Item -ItemType Directory -Path $HfHostDir -Force | Out-Null }

if ($UseGPU) {
    Write-Host "  [*] Starting with GPU acceleration..." -ForegroundColor Green
    docker run -d `
        --name pdf-chatbot-v2 `
        --gpus all `
        -p 8000:8000 `
        -p 11434:11434 `
        -v "${VectorstoreFull}:/app/vectorstore" `
        -v "${OllamaHostDir}:/root/.ollama" `
        -v "${HfHostDir}:/root/.cache/huggingface" `
        --restart unless-stopped `
        pdf-chatbot-v2:latest
} else {
    Write-Host "  [*] Starting in CPU-only mode..." -ForegroundColor Yellow
    docker run -d `
        --name pdf-chatbot-v2 `
        -p 8000:8000 `
        -p 11434:11434 `
        -e CUDA_VISIBLE_DEVICES="" `
        -e OLLAMA_NUM_GPU=0 `
        -v "${VectorstoreFull}:/app/vectorstore" `
        -v "${OllamaHostDir}:/root/.ollama" `
        -v "${HfHostDir}:/root/.cache/huggingface" `
        --restart unless-stopped `
        pdf-chatbot-v2:latest
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "  [ERROR] Failed to start Docker container." -ForegroundColor Red
    Exit 1
}

# 7e. Wait for services to come up
Write-Host "  [*] Waiting for services (Ollama model pull + FastAPI startup)..." -ForegroundColor Yellow
$WebReady = $false
$MaxWait = 600
$Waited = 0

while ($Waited -lt $MaxWait -and -not $WebReady) {
    Start-Sleep -Seconds 5
    $Waited += 5

    # Check container still running
    $Status = docker inspect --format='{{.State.Status}}' pdf-chatbot-v2 2>$null
    if ($Status -ne "running") {
        Write-Host "  [ERROR] Container stopped unexpectedly. Logs:" -ForegroundColor Red
        docker logs --tail 30 pdf-chatbot-v2
        Exit 1
    }

    # Check web server
    try {
        $resp = Invoke-WebRequest -Uri "http://localhost:8000/api/health" -Method Get -TimeoutSec 3 -ErrorAction SilentlyContinue
        if ($resp.StatusCode -eq 200) { $WebReady = $true }
    } catch {}

    if (-not $WebReady) {
        Write-Host "  ... starting (${Waited}s elapsed — model download & index loading)..." -ForegroundColor Gray
    }
}

# 7f. Final status
Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
if ($WebReady) {
    Write-Host "  ✅ PDF-Chatbot-V2 is READY!" -ForegroundColor Green
} else {
    Write-Host "  ⏳ Container is still starting (model pull may take a while)." -ForegroundColor Yellow
    Write-Host "     Monitor progress: docker logs -f pdf-chatbot-v2" -ForegroundColor Yellow
}
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  🌐 Web UI:       http://localhost:8000" -ForegroundColor Cyan
Write-Host "  🤖 Ollama API:   http://localhost:11434" -ForegroundColor Cyan
Write-Host "  📋 View logs:    docker logs -f pdf-chatbot-v2" -ForegroundColor Gray
Write-Host "  🛑 Stop:         docker stop pdf-chatbot-v2" -ForegroundColor Gray
Write-Host "  🔄 Restart:      docker restart pdf-chatbot-v2" -ForegroundColor Gray
Write-Host ""

# 7g. Open browser
if (-not $NoBrowser) {
    Write-Host "  [*] Opening browser..." -ForegroundColor Yellow
    Start-Process "http://localhost:8000"
}
