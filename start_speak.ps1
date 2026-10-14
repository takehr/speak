[CmdletBinding()]
param(
    [ValidateRange(1, 60)]
    [int]$RestartDelaySeconds = 3,

    [ValidateRange(-1, 255)]
    [int]$MicIndex = -1
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv was not found. Install uv or add it to PATH before starting Speak."
}

$logDirectory = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$launcherLog = Join-Path $logDirectory "launcher.log"

$speakArguments = @(
    "run", "python", "speak.py",
    "--no-auto-start",
    "--wake-word", "gemini",
    "--no-auto-start-wake-word", "hello",
    "--exit-word", "hello",
    "--mimic-wake-word", "copy mode",
    "--translate-wake-word", "translate",
    "--stt-model-path", ".\vosk-model-small-en-us-0.15",
    "--strict-turns",
    "--mode", "none",
    "--wake-min-confidence", "0.85",
    "--wake-min-active-seconds", "0.30",
    "--scheduled-response-timeout", "60",
    "--daily-practice-min-seconds", "60",
    "--model", "models/gemini-2.5-flash-native-audio-latest",
    "--no-websocket-proxy",
    "--live-ipv4"
)

if ($MicIndex -ge 0) {
    $speakArguments += @("--mic-index", $MicIndex.ToString())
}

while ($true) {
    $startedAt = Get-Date
    $startMessage = "{0:o} starting Speak" -f $startedAt
    Add-Content -LiteralPath $launcherLog -Value $startMessage
    Write-Host "[launcher] starting Speak"

    $exitCode = 1
    try {
        & uv @speakArguments
        $exitCode = $LASTEXITCODE
    }
    catch {
        $errorMessage = "{0:o} launcher error: {1}" -f (Get-Date), $_.Exception.Message
        Add-Content -LiteralPath $launcherLog -Value $errorMessage
        Write-Warning $errorMessage
    }

    $exitMessage = "{0:o} Speak exited with code {1}" -f (Get-Date), $exitCode
    Add-Content -LiteralPath $launcherLog -Value $exitMessage
    Write-Host "[launcher] Speak exited with code $exitCode"

    # q exits normally. Ctrl+C commonly appears as 130 or the Windows
    # STATUS_CONTROL_C_EXIT signed value. Do not restart those cases.
    if ($exitCode -in @(0, 130, -1073741510)) {
        Write-Host "[launcher] stopped normally"
        break
    }

    Write-Warning "Speak stopped unexpectedly. Restarting in $RestartDelaySeconds seconds."
    Start-Sleep -Seconds $RestartDelaySeconds
}
