# Deploy RAAAAAG !! to the Oracle Cloud VM, from PowerShell.
#
#     .\deploy\deploy.ps1
#
# Native PowerShell rather than a bash wrapper: Git Bash rewrites $HOME and
# mangles Windows paths passed to ssh/scp, which is exactly what broke the
# first attempt.

param(
    [string]$VmUser   = "ubuntu",
    [string]$VmHost   = "140.238.224.158",
    [string]$SshKey   = "$env:USERPROFILE\.ssh\hhg_vm",
    [string]$RemoteDir = ""
)

$ErrorActionPreference = "Stop"
if (-not $RemoteDir) { $RemoteDir = "/home/$VmUser/raaaaag" }

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Target   = "$VmUser@$VmHost"
$SshOpts  = @("-i", $SshKey, "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15")

function Say  { param($m) Write-Host "`n> $m" -ForegroundColor Yellow }
function Die  { param($m) Write-Host "`nX $m" -ForegroundColor Red; exit 1 }
function Remote { param($cmd) & ssh @SshOpts $Target $cmd }

# ── 1. preflight ────────────────────────────────────────────────────
Say "Checking connection to $VmHost"
if (-not (Test-Path $SshKey)) { Die "SSH key not found at $SshKey" }
Write-Host "  key: $SshKey"

$probe = Remote "echo ok"
if ($LASTEXITCODE -ne 0) {
    Die "Cannot SSH to $Target - check the key is on the VM and port 22 is open"
}

$arch  = (Remote "uname -m").Trim()
$cores = (Remote "nproc").Trim()
# free -h rather than awk: PowerShell's escaping of $2 through ssh into awk is
# a reliable source of quoting bugs for no benefit here.
$mem   = (Remote "free -h | grep Mem | tr -s ' ' | cut -d' ' -f2").Trim()
Write-Host "  arch=$arch cores=$cores ram=$mem"

$EnvFile = Join-Path $RepoRoot "backend\.env"
if (-not (Test-Path $EnvFile)) { Die "backend\.env not found - API keys are required" }

$IndexDir = Join-Path $RepoRoot "data\index"
if (-not (Get-ChildItem -Path $IndexDir -Filter *.faiss -ErrorAction SilentlyContinue)) {
    Die "No FAISS index in data\index - run: python scripts\ingest.py"
}

# ── 2. docker ───────────────────────────────────────────────────────
Say "Ensuring Docker is installed"
Remote "command -v docker >/dev/null 2>&1" | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  installing..."
    Remote "curl -fsSL https://get.docker.com | sudo sh && sudo usermod -aG docker `$USER"
}

# ── 3. firewall ─────────────────────────────────────────────────────
# Oracle images place a REJECT rule early in the INPUT chain, so opening the
# port in the VCN Security List alone is not sufficient.
Say "Opening port 80 on the instance"
Remote "sudo iptables -C INPUT -p tcp --dport 80 -j ACCEPT 2>/dev/null || sudo iptables -I INPUT 6 -p tcp --dport 80 -j ACCEPT" | Out-Null
Remote "sudo netfilter-persistent save 2>/dev/null || sudo sh -c 'iptables-save > /etc/iptables/rules.v4' 2>/dev/null" | Out-Null
Write-Host "  also add an ingress rule for TCP 80 in the Oracle VCN Security List"

# ── 4. upload ───────────────────────────────────────────────────────
Say "Uploading source"
Remote "mkdir -p $RemoteDir/data/index" | Out-Null

# Staged into a temp dir so only what the image needs is transferred: no
# venv, node_modules, .next or .git.
$Stage = Join-Path $env:TEMP "raaaaag-deploy"
if (Test-Path $Stage) { Remove-Item -Recurse -Force $Stage }
New-Item -ItemType Directory -Path $Stage -Force | Out-Null

foreach ($rel in @("backend\app", "backend\scripts", "frontend\app",
                   "frontend\components", "frontend\lib", "frontend\public", "deploy")) {
    $dest = Join-Path $Stage $rel
    New-Item -ItemType Directory -Path (Split-Path -Parent $dest) -Force | Out-Null
    Copy-Item -Recurse -Force (Join-Path $RepoRoot $rel) $dest
}
foreach ($rel in @("backend\requirements.txt", "backend\Dockerfile", "backend\pytest.ini",
                   "frontend\package.json", "frontend\package-lock.json",
                   "frontend\next.config.ts", "frontend\tsconfig.json",
                   "frontend\postcss.config.mjs", "frontend\Dockerfile")) {
    $src = Join-Path $RepoRoot $rel
    if (Test-Path $src) {
        $dest = Join-Path $Stage $rel
        New-Item -ItemType Directory -Path (Split-Path -Parent $dest) -Force | Out-Null
        Copy-Item -Force $src $dest
    }
}
Get-ChildItem -Recurse -Force $Stage -Include "__pycache__", "*.pyc", ".next", "node_modules" `
    -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

& scp @SshOpts -r -q "$Stage\*" "${Target}:$RemoteDir/"
if ($LASTEXITCODE -ne 0) { Die "Source upload failed" }
Remove-Item -Recurse -Force $Stage

Say "Uploading FAISS index"
$indexMb = [math]::Round((Get-ChildItem $IndexDir | Measure-Object Length -Sum).Sum / 1MB, 1)
Write-Host "  ${indexMb}MB"
& scp @SshOpts -q "$IndexDir\*" "${Target}:$RemoteDir/data/index/"
if ($LASTEXITCODE -ne 0) { Die "Index upload failed" }

Say "Uploading secrets"
& scp @SshOpts -q $EnvFile "${Target}:$RemoteDir/.env"
Remote "chmod 600 $RemoteDir/.env; grep -q PUBLIC_HOST $RemoteDir/.env || echo 'PUBLIC_HOST=$VmHost' >> $RemoteDir/.env" | Out-Null

# ── 5. build and start ──────────────────────────────────────────────
Say "Building images (first run takes 10-20 min on ARM)"
Remote "cd $RemoteDir/deploy && sudo docker compose --env-file ../.env build 2>&1 | tail -25"

Say "Starting stack"
Remote "cd $RemoteDir/deploy && sudo docker compose --env-file ../.env up -d"

# ── 6. wait for health ──────────────────────────────────────────────
Say "Waiting for the API"
$healthy = $false
foreach ($i in 1..40) {
    Remote "curl -sf --max-time 5 http://localhost/api/health >/dev/null 2>&1" | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  healthy after ~$($i * 5)s"
        $healthy = $true
        break
    }
    Start-Sleep -Seconds 5
}
if (-not $healthy) {
    Remote "cd $RemoteDir/deploy && sudo docker compose logs --tail=40 api"
    Die "API did not become healthy - logs above"
}

Say "Deployed"
Remote "curl -s http://localhost/api/health"

$dashed = $VmHost.Replace(".", "-")
Write-Host @"

  http://$VmHost

  Voice input will NOT work on plain HTTP - browsers block microphone
  access on insecure origins. To enable it, edit deploy/Caddyfile and
  replace ":80" with $dashed.sslip.io, then re-run this script.
  Caddy provisions the certificate automatically.

  logs:    ssh -i $SshKey $Target 'cd $RemoteDir/deploy && sudo docker compose logs -f'
  restart: ssh -i $SshKey $Target 'cd $RemoteDir/deploy && sudo docker compose restart'

"@ -ForegroundColor Green
