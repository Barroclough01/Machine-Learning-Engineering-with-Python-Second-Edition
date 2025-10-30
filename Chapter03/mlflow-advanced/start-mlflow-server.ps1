#!/usr/bin/env pwsh
# Start an MLflow tracking server (Windows PowerShell)
#
# Usage (PowerShell):
#   # Activate your conda env first (example, adapt path/name):
#   conda activate mlewp-chapter03
#   # Run this script
#   .\start-mlflow-server.ps1
#
# This script ensures artifact and DB directories exist and then starts
# the MLflow server with Windows-friendly URIs.

param(
    [string]$Port = '8000',
    [string]$ServerHost = '0.0.0.0'
)

# Set absolute paths (adjust if you want a different location)
$repoRoot = Split-Path -Parent $PSScriptRoot
$artifactRoot = Join-Path $repoRoot 'artifacts'
$dbDir = Join-Path $repoRoot 'mlflow-db'
$dbPath = Join-Path $dbDir 'mlflow.db'

# Create directories if they don't exist
New-Item -ItemType Directory -Path $artifactRoot -Force | Out-Null
New-Item -ItemType Directory -Path $dbDir -Force | Out-Null

# Convert to file URIs for mlflow (ensure slashes are forward)
$artifactUri = "file:///" + ($artifactRoot -replace '\\','/')
$dbUri = "sqlite:///" + ($dbPath -replace '\\','/')

Write-Host "Starting MLflow server"
Write-Host "  backend store: $dbUri"
Write-Host "  artifact root: $artifactUri"
Write-Host "  host: $ServerHost port: $Port"

# Run the server. If you want to run in background, remove --host/--port echo and use Start-Process.
mlflow server `
    --backend-store-uri $dbUri `
    --default-artifact-root $artifactUri `
    --host $ServerHost `
    --port $Port
