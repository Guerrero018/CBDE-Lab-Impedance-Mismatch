# Install pgvector into PostgreSQL 17
# Si no eres Admin, se reabre elevado automáticamente.

$ErrorActionPreference = "Stop"

function Test-IsAdmin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $p = New-Object Security.Principal.WindowsPrincipal($id)
    return $p.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-IsAdmin)) {
    Write-Host "Se necesitan privilegios de Administrador. Relanzando elevado..."
    $script = $MyInvocation.MyCommand.Path
    Start-Process powershell.exe -Verb RunAs -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$script`""
    ) | Out-Null
    exit 0
}

$pg = "C:\Program Files\PostgreSQL\17"
$src = Join-Path $env:TEMP "pgvector_pg17"
$zip = Join-Path $env:TEMP "pgvector_pg17.zip"
$url = "https://github.com/andreiramani/pgvector_pgsql_windows/releases/download/0.8.2_17.6/vector.v0.8.2-pg17.zip"

if (-not (Test-Path $pg)) {
    Write-Error "No se encontró PostgreSQL en: $pg"
}

if (-not (Test-Path "$src\lib\vector.dll")) {
    Write-Host "Descargando pgvector..."
    Invoke-WebRequest -Uri $url -OutFile $zip
    if (Test-Path $src) { Remove-Item $src -Recurse -Force }
    Expand-Archive -Path $zip -DestinationPath $src -Force
}

Write-Host "Copiando a $pg ..."
Copy-Item "$src\lib\vector.dll" "$pg\lib\" -Force
Copy-Item "$src\share\extension\*" "$pg\share\extension\" -Force

Write-Host ""
Write-Host "Archivos instalados:"
Write-Host "  $(Test-Path "$pg\lib\vector.dll")  $pg\lib\vector.dll"
Write-Host "  $(Test-Path "$pg\share\extension\vector.control")  $pg\share\extension\vector.control"
Write-Host ""
Write-Host "Siguiente paso:"
Write-Host "  1) Reinicia el servicio: Restart-Service postgresql-x64-17"
Write-Host "     (el nombre exacto puede variar; mira Get-Service *postgres*)"
Write-Host "  2) En psql / Python: CREATE EXTENSION IF NOT EXISTS vector;"
Write-Host "  3) python g0.py"
Write-Host ""
Read-Host "Pulsa Enter para cerrar"
