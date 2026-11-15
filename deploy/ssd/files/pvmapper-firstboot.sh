#!/bin/bash
# PV-Mapper first-boot setup on the target PC (runs as root via pvmapper-firstboot.service).
# Installs PyTorch (ROCm), PV-Mapper and the model weights, checks the GPU, then starts the web UI.
# While it runs, a status page is served on port 8765. Idempotent: if a step fails, the next boot retries.
set -uo pipefail

APP=/opt/pvmapper
LOG=/var/log/pvmapper-firstboot.log
STATUS=$APP/setup
REPO="${PVMAPPER_REPO:-https://github.com/chri6o/deeppvmapper}"
BRANCH="${PVMAPPER_BRANCH:-pvmapper-eu}"
ROCM_INDEX="${PVMAPPER_ROCM_INDEX:-https://download.pytorch.org/whl/rocm7.2}"
USER_NAME=pvmapper
[ -f /etc/default/pvmapper-setup ] && . /etc/default/pvmapper-setup

mkdir -p "$STATUS"
exec >>"$LOG" 2>&1
echo "=== $(date '+%F %T') PV-Mapper Einrichtung startet ==="

STEP="Start"
render() {
  local state="$1"
  {
    echo '<!doctype html><html lang="de"><meta charset="utf-8"><meta http-equiv="refresh" content="5">'
    echo '<title>PV-Mapper wird eingerichtet</title><body style="font:15px system-ui;max-width:900px;margin:30px auto;padding:0 16px">'
    echo "<h2>PV-Mapper wird eingerichtet …</h2><p><b>Schritt:</b> $STEP &nbsp; <b>Status:</b> $state</p>"
    echo '<p>Beim ersten Start werden PyTorch für die Radeon (~4 GB) und die Modelle (~630 MB) geladen – je nach Leitung 10–30 Minuten. Die Seite aktualisiert sich selbst; danach erscheint hier PV-Mapper.</p>'
    echo '<pre style="background:#0f172a;color:#d6deeb;padding:10px;border-radius:6px;white-space:pre-wrap;font-size:12px">'
    tail -n 40 "$LOG" | sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g'
    echo '</pre></body></html>'
  } > "$STATUS/index.html.tmp" && mv "$STATUS/index.html.tmp" "$STATUS/index.html"
}
step() { STEP="$1"; echo; echo ">>> $1"; render "läuft"; }
fail() { echo "FEHLER in Schritt: $STEP"; render "FEHLER – nächster Versuch beim nächsten Neustart (oder: sudo systemctl restart pvmapper-firstboot)"; sleep 2; kill "$SRV" 2>/dev/null; exit 1; }

render "läuft"
python3 -m http.server 8765 --directory "$STATUS" >/dev/null 2>&1 &
SRV=$!
( while kill -0 $$ 2>/dev/null; do render "läuft"; sleep 5; done ) &

step "Warte auf Internet"
for i in $(seq 1 120); do curl -sfI https://github.com >/dev/null && break; sleep 5; done
curl -sfI https://github.com >/dev/null || fail

step "Systempakete"
export DEBIAN_FRONTEND=noninteractive
apt-get update -q || fail
apt-get install -y -q git python3-venv python3-pip avahi-daemon curl ca-certificates || fail
id "$USER_NAME" >/dev/null 2>&1 || useradd -m -s /bin/bash "$USER_NAME"
usermod -aG render,video "$USER_NAME"
mkdir -p "$APP/models" "$APP/jobs"

step "PV-Mapper holen ($BRANCH)"
if [ -d "$APP/app/.git" ]; then
  git -C "$APP/app" fetch -q origin "$BRANCH" && git -C "$APP/app" checkout -q "$BRANCH" && git -C "$APP/app" reset -q --hard "origin/$BRANCH" || fail
else
  git clone -q --depth 1 -b "$BRANCH" "$REPO" "$APP/app" || fail
fi

step "Python-Umgebung und PyTorch für AMD ROCm (großer Download)"
[ -x "$APP/venv/bin/python" ] || python3 -m venv "$APP/venv" || fail
"$APP/venv/bin/pip" install -q --upgrade pip || fail
"$APP/venv/bin/python" -c "import torch, sys; sys.exit(0 if getattr(torch.version,'hip',None) else 1)" 2>/dev/null \
  || "$APP/venv/bin/pip" install --progress-bar off torch torchvision --index-url "$ROCM_INDEX" || fail

step "Weitere Pakete"
"$APP/venv/bin/pip" install -q -r "$APP/app/requirements-pvmapper.txt" pyogrio || fail

step "Modellgewichte laden"
chown -R "$USER_NAME:$USER_NAME" "$APP"
cd "$APP" && sudo -u "$USER_NAME" PYTHONPATH="$APP/app" "$APP/venv/bin/python" -m pvmapper --config "$APP/pvmapper.yml" models || fail

step "Grafikkarte prüfen"
GPUTEST='import torch,sys
ok=torch.cuda.is_available()
if ok:
    x=torch.randn(512,512,device="cuda"); (x@x).sum().item()
    print("GPU:",torch.cuda.get_device_name(0),"| ROCm",torch.version.hip)
sys.exit(0 if ok else 1)'
: > /etc/default/pvmapper
if sudo -u "$USER_NAME" "$APP/venv/bin/python" -c "$GPUTEST"; then
  echo "Grafikkarte läuft ohne Zusatzeinstellung."
elif sudo -u "$USER_NAME" HSA_OVERRIDE_GFX_VERSION=11.0.0 "$APP/venv/bin/python" -c "$GPUTEST"; then
  echo "HSA_OVERRIDE_GFX_VERSION=11.0.0" > /etc/default/pvmapper
  echo "Grafikkarte läuft mit HSA_OVERRIDE_GFX_VERSION=11.0.0 (gespeichert)."
else
  echo "WARNUNG: keine GPU nutzbar – PV-Mapper läuft auf der CPU (langsam). Details: lspci | grep -i vga; ls /dev/kfd"
fi

step "Fertig – starte PV-Mapper"
chown -R "$USER_NAME:$USER_NAME" "$APP"
chmod 600 "$APP/pvmapper.yml"
touch "$APP/.installed"
kill "$SRV" 2>/dev/null; sleep 1
systemctl daemon-reload
systemctl enable --now pvmapper.service
systemctl disable pvmapper-firstboot.service
echo "=== $(date '+%F %T') Einrichtung abgeschlossen: http://$(hostname).local:8765 ==="
exit 0
