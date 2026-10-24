param([switch]$Cpu, [switch]$InstallOnly, [switch]$Check)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = '1'
$projectRoot = $PSScriptRoot
Set-Location -LiteralPath $projectRoot

function Run-Python {
    param([string[]]$PythonArgs)
    & $script:projectPython @PythonArgs
    if ($LASTEXITCODE -ne 0) { throw "Python завершился с кодом $LASTEXITCODE." }
}

try {
    $runtimeRoot = Join-Path $projectRoot '.runtime'
    $projectPython = Join-Path $runtimeRoot 'python.exe'
    $localPythonFile = Join-Path $projectRoot '.local-python.txt'
    if (Test-Path -LiteralPath $localPythonFile) {
        $candidate = (Get-Content -LiteralPath $localPythonFile -Raw).Trim()
        if (Test-Path -LiteralPath $candidate) { $projectPython = $candidate }
    }
    if (-not (Test-Path -LiteralPath $projectPython)) {
        Write-Host 'Устанавливается отдельный Python 3.12. Системный Python не меняется.'
        New-Item -ItemType Directory -Force -Path $runtimeRoot | Out-Null
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        $pythonZip = Join-Path $runtimeRoot 'python.zip'
        Invoke-WebRequest -UseBasicParsing -Uri 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip' -OutFile $pythonZip
        Expand-Archive -LiteralPath $pythonZip -DestinationPath $runtimeRoot -Force
        $pthFile = Join-Path $runtimeRoot 'python312._pth'
        # Добавляем корень проекта для импортов app/predict/evaluate в embedded Python.
        [IO.File]::WriteAllText($pthFile, "python312.zip`r`n.`r`n..`r`nimport site`r`n", [Text.Encoding]::ASCII)
        $getPip = Join-Path $runtimeRoot 'get-pip.py'
        Invoke-WebRequest -UseBasicParsing -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile $getPip
        Run-Python -PythonArgs @($getPip, '--no-warn-script-location')
    }
    $marker = Join-Path (Split-Path -Parent $projectPython) 'text-project-ready-v3.txt'
    if (-not (Test-Path -LiteralPath $marker)) {
        Write-Host 'Устанавливаются библиотеки. Нужен интернет, оплата не требуется.'
        $useGpu = (-not $Cpu) -and ($null -ne (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue))
        if ($useGpu) { $torchIndex = 'https://download.pytorch.org/whl/cu118' }
        else { $torchIndex = 'https://download.pytorch.org/whl/cpu' }
        try {
            Run-Python -PythonArgs @('-m', 'pip', 'install', '--disable-pip-version-check', '--no-cache-dir', 'torch==2.7.1', 'torchvision==0.22.1', '--index-url', $torchIndex)
        } catch {
            if (-not $useGpu) { throw }
            Write-Host 'Не удалось установить ускорение NVIDIA. Устанавливается вариант для процессора.'
            Run-Python -PythonArgs @('-m', 'pip', 'install', '--disable-pip-version-check', '--no-cache-dir', 'torch==2.7.1', 'torchvision==0.22.1', '--index-url', 'https://download.pytorch.org/whl/cpu')
        }
        Run-Python -PythonArgs @('-m', 'pip', 'install', '--disable-pip-version-check', '--no-cache-dir', '-r', (Join-Path $projectRoot 'requirements-lock.txt'))
        Run-Python -PythonArgs @('-c', "import torch, transformers, PIL; print('Libraries ready')")
        [IO.File]::WriteAllText($marker, 'ready', [Text.Encoding]::ASCII)
    }
    Run-Python -PythonArgs @('-m', 'support.download_models')
    if ($Check) {
        Run-Python -PythonArgs @('-m', 'unittest', 'discover', '-s', 'tests', '-v')
        Run-Python -PythonArgs @('-c', "import torch; print('Torch:',torch.__version__); print('GPU:',torch.cuda.is_available())")
    }
    if (-not $InstallOnly) {
        Write-Host 'Программа запускается. Оставьте это окно открытым.'
        if ($Cpu) { Run-Python -PythonArgs @((Join-Path $projectRoot 'app.py'), '--device', 'cpu') }
        else { Run-Python -PythonArgs @((Join-Path $projectRoot 'app.py')) }
    }
} catch {
    Write-Host "Ошибка: $_" -ForegroundColor Red
    exit 1
}
