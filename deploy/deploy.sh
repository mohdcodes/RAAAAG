#!/usr/bin/env bash
#
# Deploy RAAAAAG !! to the Oracle Cloud VM.
#
# Run from the repo root on your own machine:
#     bash deploy/deploy.sh
#
# What it does:
#   1. checks the VM is reachable and has Docker
#   2. uploads source, the prebuilt FAISS index, and secrets
#   3. builds and starts the stack there
#   4. waits for health and reports the URL
#
# The index is uploaded rather than rebuilt: embedding on a cloud CPU would
# take hours to reproduce something that already exists locally.

set -euo pipefail

VM_USER="${VM_USER:-ubuntu}"
VM_HOST="${VM_HOST:-140.238.224.158}"
REMOTE_DIR="${REMOTE_DIR:-/home/$VM_USER/raaaaag}"

# Locate the SSH key. Under Git Bash on Windows, $HOME is often /home/<user>
# rather than the Windows profile, so ~/.ssh does not resolve to where
# ssh-keygen actually wrote the key. Check the likely locations instead of
# assuming one.
find_key() {
  local candidates=(
    "${SSH_KEY:-}"
    "$HOME/.ssh/hhg_vm"
    "${USERPROFILE:-}/.ssh/hhg_vm"
    "/c/Users/${USERNAME:-$USER}/.ssh/hhg_vm"
    "/mnt/c/Users/${USERNAME:-$USER}/.ssh/hhg_vm"
  )
  for candidate in "${candidates[@]}"; do
    [[ -n "$candidate" ]] || continue
    # Windows paths arrive with backslashes; bash test needs forward slashes.
    candidate="${candidate//\\//}"
    [[ -f "$candidate" ]] && { printf '%s' "$candidate"; return 0; }
  done
  return 1
}

SSH_KEY="$(find_key)" || {
  printf '\n\033[1;31m✗ SSH key not found.\033[0m\n' >&2
  printf 'Looked in: $HOME/.ssh, %%USERPROFILE%%/.ssh, /c/Users/$USERNAME/.ssh\n' >&2
  printf 'Pass it explicitly:  SSH_KEY=/c/Users/you/.ssh/hhg_vm bash deploy/deploy.sh\n\n' >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)

say()  { printf '\n\033[1;33m▸ %s\033[0m\n' "$*"; }
fail() { printf '\n\033[1;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

remote() { ssh "${SSH_OPTS[@]}" "$VM_USER@$VM_HOST" "$@"; }

# ── 1. preflight ────────────────────────────────────────────────────
say "Checking connection to $VM_HOST"
echo "  key: $SSH_KEY"

# OpenSSH refuses a key that is group/world readable. Windows-created keys
# frequently are, and the resulting error names permissions rather than the
# fix, so tighten it here.
chmod 600 "$SSH_KEY" 2>/dev/null || true

remote "echo ok" >/dev/null 2>&1 || fail \
  "Cannot SSH to $VM_USER@$VM_HOST — check the key is installed on the VM and port 22 is open"

ARCH="$(remote 'uname -m')"
CORES="$(remote 'nproc')"
MEM="$(remote "free -g | awk '/^Mem:/{print \$2}'")"
echo "  arch=$ARCH cores=$CORES ram=${MEM}GB"

[[ -f "$REPO_ROOT/backend/.env" ]] || fail "backend/.env not found — API keys are required"

INDEX_DIR="$REPO_ROOT/data/index"
compgen -G "$INDEX_DIR/*.faiss" >/dev/null \
  || fail "No FAISS index in data/index — run: python scripts/ingest.py"

# ── 2. docker ───────────────────────────────────────────────────────
say "Ensuring Docker is installed"
if ! remote 'command -v docker' >/dev/null 2>&1; then
  echo "  installing…"
  remote 'curl -fsSL https://get.docker.com | sudo sh && sudo usermod -aG docker $USER'
  echo "  installed (group change applies on next login; sudo is used below)"
fi

# ── 3. firewall ─────────────────────────────────────────────────────
# Oracle images ship with a REJECT rule early in the INPUT chain, so opening
# the port in the cloud Security List alone is not enough.
say "Opening port 80 on the instance"
remote 'sudo iptables -C INPUT -p tcp --dport 80 -j ACCEPT 2>/dev/null || \
        sudo iptables -I INPUT 6 -p tcp --dport 80 -j ACCEPT' || true
remote 'sudo netfilter-persistent save 2>/dev/null || \
        sudo sh -c "iptables-save > /etc/iptables/rules.v4" 2>/dev/null' || true
echo "  note: also allow ingress on TCP 80 in the Oracle VCN Security List"

# ── 4. upload ───────────────────────────────────────────────────────
say "Uploading source"
remote "mkdir -p $REMOTE_DIR/data/index"

tar czf - -C "$REPO_ROOT" \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='venv' \
    --exclude='node_modules' --exclude='.next' --exclude='.git' \
    backend/app backend/scripts backend/requirements.txt \
    backend/Dockerfile backend/pytest.ini \
    frontend/app frontend/components frontend/lib frontend/public \
    frontend/package.json frontend/package-lock.json \
    frontend/next.config.ts frontend/tsconfig.json \
    frontend/postcss.config.mjs frontend/Dockerfile \
    deploy \
  | remote "tar xzf - -C $REMOTE_DIR"

say "Uploading FAISS index"
INDEX_MB=$(du -sm "$INDEX_DIR" | cut -f1)
echo "  ${INDEX_MB}MB"
scp "${SSH_OPTS[@]}" -q "$INDEX_DIR"/* "$VM_USER@$VM_HOST:$REMOTE_DIR/data/index/"

say "Uploading secrets"
# Piped over the existing SSH channel rather than scp'd, so the keys never
# land in a temp file anywhere.
remote "cat > $REMOTE_DIR/.env && chmod 600 $REMOTE_DIR/.env" \
  < "$REPO_ROOT/backend/.env"
remote "grep -q PUBLIC_HOST $REMOTE_DIR/.env || echo 'PUBLIC_HOST=$VM_HOST' >> $REMOTE_DIR/.env"

# ── 5. build and start ──────────────────────────────────────────────
say "Building images (first run takes 10-20 min on ARM)"
remote "cd $REMOTE_DIR/deploy && sudo docker compose --env-file ../.env build 2>&1 | tail -20"

say "Starting stack"
remote "cd $REMOTE_DIR/deploy && sudo docker compose --env-file ../.env up -d"

# ── 6. wait for health ──────────────────────────────────────────────
say "Waiting for the API to come up"
for i in $(seq 1 40); do
  if remote "curl -sf --max-time 5 http://localhost/api/health" >/dev/null 2>&1; then
    echo "  healthy after ~$((i * 5))s"
    break
  fi
  [[ $i -eq 40 ]] && {
    remote "cd $REMOTE_DIR/deploy && sudo docker compose logs --tail=40 api"
    fail "API did not become healthy — logs above"
  }
  sleep 5
done

say "Deployed"
remote "curl -s http://localhost/api/health" | head -c 400
cat <<EOF


  http://$VM_HOST

  Voice input will NOT work on plain HTTP — browsers block microphone
  access on insecure origins. To enable it, edit deploy/Caddyfile and
  replace ":80" with a hostname (e.g. ${VM_HOST//./-}.sslip.io), then
  re-run this script. Caddy provisions the certificate automatically.

  logs:    ssh -i $SSH_KEY $VM_USER@$VM_HOST 'cd $REMOTE_DIR/deploy && sudo docker compose logs -f'
  restart: ssh -i $SSH_KEY $VM_USER@$VM_HOST 'cd $REMOTE_DIR/deploy && sudo docker compose restart'

EOF
