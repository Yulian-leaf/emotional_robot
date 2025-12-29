Param(
  [Alias('Host')]
  [string]$BindHost = "127.0.0.1",
  [int]$Port = 3000
)

$ErrorActionPreference = "Stop"

$frontendDir = Join-Path $PSScriptRoot "..\frontend"
$frontendDir = (Resolve-Path $frontendDir).Path

# Ensure Node/NPM are discoverable even in conda shells.
$nodeDir = Join-Path $env:ProgramFiles "nodejs"
$npmDir = Join-Path $env:APPDATA "npm"
if (-not (Test-Path $nodeDir)) {
  throw "Node.js not found at '$nodeDir'. Please install Node.js or add it to PATH."
}
$env:Path = "$nodeDir;$npmDir;$env:Path"

function Stop-PortListener([int]$p) {
  $lines = netstat -ano | Select-String ":$p\s+.*LISTENING\s+(\d+)" | ForEach-Object { $_.Line }
  foreach ($line in $lines) {
    if ($line -match "LISTENING\s+(\d+)$") {
      $listenerPid = [int]$Matches[1]
      try {
        Stop-Process -Id $listenerPid -Force -ErrorAction Stop
        Write-Host "Stopped PID $listenerPid listening on $p" -ForegroundColor Yellow
      }
      catch {
        Write-Warning "Failed to stop PID $listenerPid on port ${p}: $($_.Exception.Message)"
      }
    }
  }
}

Stop-PortListener -p $Port

Write-Host "Starting frontend (Vite) on http://${BindHost}`:${Port}/" -ForegroundColor Green

& "$nodeDir\npm.cmd" --prefix $frontendDir run dev -- --host $BindHost --port $Port --strictPort
