# run.ps1 — первый RAG-запрос (день 22).
#
#   .\run.ps1                                      # 10 контрольных вопросов: без RAG vs с RAG
#   .\run.ps1 -Mode ask -Query "Какие слои памяти в дне 11?"   # один вопрос, оба ответа рядом
#   .\run.ps1 -Mode chat                           # диалог: /rag, /plain, /both, /sources
#   .\run.ps1 -Mode index                          # пересобрать индекс (после изменения docs\)
#
# Ключ AITunnel скрипт спросит со скрытым вводом. Модель по умолчанию — gpt-5.6-luna.
# Если Windows блокирует запуск:  powershell -ExecutionPolicy Bypass -File .\run.ps1

param(
    [ValidateSet("eval", "ask", "chat", "index")]
    [string]$Mode = "eval",
    [string]$Model = "gpt-5.6-luna",
    [ValidateSet("aitunnel", "local")]
    [string]$Backend = "aitunnel",
    [string]$Query = ""
)

Set-Location $PSScriptRoot

function Find-Python {
    foreach ($candidate in @("py", "python",
        "$env:LOCALAPPDATA\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\python.exe")) {
        try {
            $output = & $candidate --version 2>&1
            if ($output -match "Python \d") { return $candidate }
        }
        catch { }
    }
    return $null
}

$py = Find-Python
if (-not $py) {
    Write-Host "Python не найден. Установите Python 3 с python.org или из Microsoft Store." -ForegroundColor Red
    exit 1
}
Write-Host "Использую Python: $py" -ForegroundColor Green

Write-Host "Устанавливаю зависимости..." -ForegroundColor Cyan
& $py -m pip install -r requirements.txt --quiet --disable-pip-version-check

if (-not $env:AITUNNEL_API_KEY -and -not ($Mode -eq "index" -and $Backend -eq "local")) {
    $secureKey = Read-Host "Введите ваш AITUNNEL_API_KEY" -AsSecureString
    $env:AITUNNEL_API_KEY = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto(
        [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
    )
}

switch ($Mode) {
    "index" { & $py .\build_index.py --backend $Backend }
    "eval"  { & $py .\evaluate.py --model $Model --backend $Backend }
    "chat"  { & $py .\agent.py --model $Model --backend $Backend }
    "ask"   {
        if (-not $Query) { Write-Host "Укажите -Query ""вопрос""" -ForegroundColor Yellow; exit 1 }
        & $py .\agent.py --model $Model --backend $Backend --mode both --ask $Query
    }
}
