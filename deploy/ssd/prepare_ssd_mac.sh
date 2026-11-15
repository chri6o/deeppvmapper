#!/bin/bash
# =============================================================================
# PV-Mapper: externe SSD am Mac als fertiges Boot-Laufwerk für den PV-Mapper-PC einrichten
#
# Was passiert:
#   1. Ubuntu Server 24.04 LTS (Installations-ISO) wird geladen und geprüft.
#   2. Eine unbeaufsichtigte Installation läuft in einer virtuellen x86-Maschine (QEMU) auf dem Mac
#      und schreibt Ubuntu direkt auf die externe SSD – inkl. Zugangsdaten, PV-Mapper-Konfiguration,
#      Netzwerk (Kabel per DHCP, optional WLAN) und einem Einrichtungsdienst für den ersten Start.
#   3. SSD in den PC, einschalten. Beim ersten Start lädt der PC PyTorch für die Radeon, PV-Mapper und
#      die Modelle (10–30 Min.); den Fortschritt zeigt  http://pvmapper.local:8765  im Browser.
#
# Aufruf (im Terminal auf dem Mac, SSD angeschlossen):
#   cd deploy/ssd && ./prepare_ssd_mac.sh
# Optional vorab Zugangsdaten in  pvmapper-ssd.conf  eintragen (Vorlage: pvmapper-ssd.conf.example),
# sonst fragt das Skript nach.
#
# Voraussetzungen: macOS (Apple Silicon oder Intel), Homebrew (https://brew.sh), ~10 GB frei,
# Internet. Auf Apple Silicon wird x86 emuliert: die Installation dauert dann 30–90 Minuten.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${PVMAPPER_SSD_WORK:-$HOME/pvmapper-ssd}"
CONF="$HERE/pvmapper-ssd.conf"
UBUNTU_BASE="https://releases.ubuntu.com/24.04"
mkdir -p "$WORK"

say()  { printf '\n\033[1;33m==> %s\033[0m\n' "$*"; }
die()  { printf '\n\033[1;31mFEHLER: %s\033[0m\n' "$*" >&2; exit 1; }
b64()  { base64 < "$1" | tr -d '\n'; }
b64s() { printf '%s' "$1" | base64 | tr -d '\n'; }

RENDER_ONLY="${PVMAPPER_SSD_RENDER_ONLY:-}"     # tests: only write the seed files, no download/disk/VM
[ -n "$RENDER_ONLY" ] || [ "$(uname)" = "Darwin" ] || die "Dieses Skript ist für macOS."

if [ -z "$RENDER_ONLY" ]; then
# ----------------------------------------------------------------------------- 1. Werkzeuge
say "Prüfe Werkzeuge"
command -v brew >/dev/null || die "Homebrew fehlt. Installation: https://brew.sh – danach Skript erneut starten."
command -v qemu-system-x86_64 >/dev/null || { echo "Installiere QEMU (brew install qemu) …"; brew install qemu; }
QEMU_SHARE="$(brew --prefix qemu)/share/qemu"
OVMF_CODE="$QEMU_SHARE/edk2-x86_64-code.fd"
OVMF_VARS_TEMPLATE="$QEMU_SHARE/edk2-i386-vars.fd"
[ -f "$OVMF_CODE" ] || die "UEFI-Firmware nicht gefunden: $OVMF_CODE"
QEMU_BIN="$(command -v qemu-system-x86_64)"
fi

# ----------------------------------------------------------------------------- 2. Zugangsdaten
[ -f "$CONF" ] && { echo "Lese $CONF"; . "$CONF"; }
ask() {           # ask VAR "Frage" [default] [secret]
  local var="$1" q="$2" def="${3:-}" secret="${4:-}" val isset
  eval "isset=\${$var+x}"
  [ -n "$isset" ] && return            # already given (conf file / environment), even if empty
  if [ -n "$secret" ]; then read -r -s -p "$q: " val; echo; else read -r -p "$q${def:+ [$def]}: " val; fi
  [ -z "$val" ] && val="$def"
  eval "$var=\$val"
}
say "Zugangsdaten und Einstellungen (Enter = Vorgabe / leer lassen)"
ask PC_HOSTNAME  "Rechnername im Netz" "pvmapper"
while [ -z "${LINUX_PASSWORD:-}" ]; do
  read -r -s -p "Passwort für den Linux-Benutzer 'pvmapper' (für SSH/sudo): " p1; echo
  read -r -s -p "Passwort wiederholen: " p2; echo
  [ -n "$p1" ] && [ "$p1" = "$p2" ] && LINUX_PASSWORD="$p1" || echo "Leer oder ungleich – bitte noch einmal."
done
ask WEB_PASSWORD "Passwort für die PV-Mapper-Oberfläche im Browser (leer = ohne)" "" secret
ask PVMAP_URL    "PV-Map-Adresse" "https://pv-map.solar"
ask PVMAP_TOKEN  "PV-Map-API-Token (pvm_…, leer = später in den Einstellungen)" "" secret
ask ANTHROPIC_API_KEY "Anthropic-API-Schlüssel für Claude (leer = später)" "" secret
ask OPENAI_API_KEY    "OpenAI-API-Schlüssel für ChatGPT (leer = später)" "" secret
ask WIFI_SSID    "WLAN-Name (leer = nur Netzwerkkabel)" ""
[ -n "${WIFI_SSID:-}" ] && ask WIFI_PASSWORD "WLAN-Passwort" "" secret
SSH_KEY="${SSH_KEY:-}"
if [ -z "$SSH_KEY" ]; then
  for k in "$HOME/.ssh/id_ed25519.pub" "$HOME/.ssh/id_rsa.pub" "$HOME/.ssh/id_ecdsa.pub"; do
    [ -f "$k" ] && { SSH_KEY="$(cat "$k")"; echo "SSH-Schlüssel des Macs wird hinterlegt: $k"; break; }
  done
fi
PC_HOSTNAME="$(printf '%s' "$PC_HOSTNAME" | tr -cd 'a-zA-Z0-9-' | tr 'A-Z' 'a-z')"
[ -n "$PC_HOSTNAME" ] || PC_HOSTNAME=pvmapper

if [ -z "$RENDER_ONLY" ]; then
# ----------------------------------------------------------------------------- 3. Ubuntu-ISO
say "Lade Ubuntu Server 24.04 LTS"
curl -fsSL "$UBUNTU_BASE/SHA256SUMS" -o "$WORK/SHA256SUMS" || die "SHA256SUMS nicht ladbar"
ISO_NAME="$(grep -o 'ubuntu-24\.04[0-9.]*-live-server-amd64\.iso' "$WORK/SHA256SUMS" | sort -V | tail -1)"
[ -n "$ISO_NAME" ] || die "kein Server-ISO in SHA256SUMS gefunden"
ISO="$WORK/$ISO_NAME"
SUM="$(grep " \*$ISO_NAME\$" "$WORK/SHA256SUMS" | cut -d' ' -f1)"
if [ ! -f "$ISO" ] || [ "$(shasum -a 256 "$ISO" | cut -d' ' -f1)" != "$SUM" ]; then
  curl -fL --progress-bar -C - "$UBUNTU_BASE/$ISO_NAME" -o "$ISO" || die "Download fehlgeschlagen"
  [ "$(shasum -a 256 "$ISO" | cut -d' ' -f1)" = "$SUM" ] || die "Prüfsumme des ISO stimmt nicht – Datei löschen und erneut starten: $ISO"
fi
echo "OK: $ISO_NAME"

say "Hole Kernel und Initrd aus dem ISO"
KDIR="$WORK/kernel"; rm -rf "$KDIR"; mkdir -p "$KDIR"
KERNEL_PATH="$(tar -tf "$ISO" | grep -E '^(\./)?casper/(hwe-)?vmlinuz$' | sort | head -1)"
INITRD_PATH="${KERNEL_PATH/vmlinuz/initrd}"
[ -n "$KERNEL_PATH" ] || die "Kernel im ISO nicht gefunden"
tar -xf "$ISO" -C "$KDIR" "$KERNEL_PATH" "$INITRD_PATH" || die "Kernel/Initrd nicht entpackbar"
fi

# ----------------------------------------------------------------------------- 4. Autoinstall-Konfiguration
say "Erzeuge Installationskonfiguration"
SEED="$WORK/seed"; rm -rf "$SEED"; mkdir -p "$SEED"
TMP="$WORK/tmp"; rm -rf "$TMP"; mkdir -p "$TMP"

yaml_q() { printf '"%s"' "$(printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g')"; }
{
  echo "# PV-Mapper configuration (written by prepare_ssd_mac.sh)"
  echo "model_dir: /opt/pvmapper/models"
  echo "jobs_dir: /opt/pvmapper/jobs"
  echo "profile: nrw"
  echo "device: auto"
  echo "batch_size: 32"
  echo "fetch_workers: 8"
  echo "pvmap_url: $(yaml_q "$PVMAP_URL")"
  echo "pvmap_token: $(yaml_q "${PVMAP_TOKEN:-}")"
  echo "anthropic_api_key: $(yaml_q "${ANTHROPIC_API_KEY:-}")"
  echo "openai_api_key: $(yaml_q "${OPENAI_API_KEY:-}")"
  echo "web_password: $(yaml_q "${WEB_PASSWORD:-}")"
} > "$TMP/pvmapper.yml"

{
  echo "network:"
  echo "  version: 2"
  echo "  ethernets:"
  echo "    alle-kabel:"
  echo "      match: {name: \"en*\"}"
  echo "      dhcp4: true"
  echo "      optional: true"
  if [ -n "${WIFI_SSID:-}" ]; then
    echo "  wifis:"
    echo "    alle-wlan:"
    echo "      match: {name: \"wl*\"}"
    echo "      dhcp4: true"
    echo "      optional: true"
    echo "      access-points:"
    echo "        $(yaml_q "$WIFI_SSID"):"
    echo "          password: $(yaml_q "${WIFI_PASSWORD:-}")"
  fi
} > "$TMP/netplan.yaml"

printf 'pvmapper:%s\n' "$LINUX_PASSWORD" > "$TMP/chpasswd"

F="$HERE/files"
B_CONF="$(b64 "$TMP/pvmapper.yml")"
B_NET="$(b64 "$TMP/netplan.yaml")"
B_PW="$(b64 "$TMP/chpasswd")"
B_FB="$(b64 "$F/pvmapper-firstboot.sh")"
B_FBS="$(b64 "$F/pvmapper-firstboot.service")"
B_SVC="$(b64 "$F/pvmapper.service")"
B_UPD="$(b64 "$F/pvmapper-update")"
rm -f "$TMP/chpasswd"

KEYS_YAML="[]"
[ -n "$SSH_KEY" ] && KEYS_YAML="[$(yaml_q "$SSH_KEY")]"

cat > "$SEED/user-data" <<EOF
#cloud-config
autoinstall:
  version: 1
  interactive-sections: []
  locale: de_DE.UTF-8
  keyboard: {layout: de}
  timezone: Europe/Berlin
  refresh-installer: {update: false}
  source: {id: ubuntu-server-minimal}
  network:
    version: 2
    ethernets:
      vmnet: {match: {name: "en*"}, dhcp4: true}
  storage:
    layout: {name: direct}
  identity:
    hostname: $PC_HOSTNAME
    username: pvmapper
    realname: PV-Mapper
    password: "\$6\$pvmapperinit\$3voY.fjqaErzM/vOdhoKkICNAzmEdIp1Wl13pDa6YS9X5zTsvFpceS92EY6duAeJN2AlBzLNH3y/25cOcRhLA0"
  ssh:
    install-server: true
    allow-pw: true
    authorized-keys: $KEYS_YAML
  codecs: {install: false}
  drivers: {install: false}
  updates: security
  packages: [avahi-daemon, git, python3-venv, python3-pip, curl, wpasupplicant, pciutils]
  late-commands:
    - mkdir -p /target/opt/pvmapper/models /target/opt/pvmapper/jobs
    - echo $B_CONF | base64 -d > /target/opt/pvmapper/pvmapper.yml
    - chmod 600 /target/opt/pvmapper/pvmapper.yml
    - rm -f /target/etc/netplan/*.yaml
    - echo $B_NET | base64 -d > /target/etc/netplan/90-pvmapper.yaml
    - chmod 600 /target/etc/netplan/90-pvmapper.yaml
    - echo $B_FB | base64 -d > /target/usr/local/sbin/pvmapper-firstboot
    - chmod 755 /target/usr/local/sbin/pvmapper-firstboot
    - echo $B_UPD | base64 -d > /target/usr/local/bin/pvmapper-update
    - chmod 755 /target/usr/local/bin/pvmapper-update
    - echo $B_FBS | base64 -d > /target/etc/systemd/system/pvmapper-firstboot.service
    - echo $B_SVC | base64 -d > /target/etc/systemd/system/pvmapper.service
    - curtin in-target -- systemctl enable pvmapper-firstboot.service avahi-daemon.service
    - curtin in-target -- sh -c 'id pvmapper || useradd -m -s /bin/bash -G sudo pvmapper'
    - curtin in-target -- usermod -aG sudo,render,video pvmapper
    - echo $B_PW | base64 -d | chroot /target chpasswd
    - curtin in-target -- chown -R pvmapper:pvmapper /opt/pvmapper
    # boot on another machine: firmware fallback path EFI/BOOT/BOOTX64.EFI (shim) in addition to the NVRAM entry
    - curtin in-target -- sh -c 'echo "grub-efi-amd64-signed grub2/force_efi_extra_removable boolean true" | debconf-set-selections; grub-install --target=x86_64-efi --efi-directory=/boot/efi --uefi-secure-boot --force-extra-removable --no-nvram || true'
    - sh -c 'E=/target/boot/efi/EFI; mkdir -p \$E/BOOT; [ -f \$E/BOOT/BOOTX64.EFI ] || { cp \$E/ubuntu/shimx64.efi \$E/BOOT/BOOTX64.EFI; cp \$E/ubuntu/grubx64.efi \$E/BOOT/; [ -f \$E/ubuntu/mmx64.efi ] && cp \$E/ubuntu/mmx64.efi \$E/BOOT/; }; ls -R \$E'
    - curtin in-target -- sh -c 'sed -i "s/^GRUB_TIMEOUT=.*/GRUB_TIMEOUT=3/; s/^GRUB_TIMEOUT_STYLE=.*/GRUB_TIMEOUT_STYLE=menu/" /etc/default/grub; update-grub'
  shutdown: poweroff
EOF
printf 'instance-id: pvmapper-ssd\nlocal-hostname: %s\n' "$PC_HOSTNAME" > "$SEED/meta-data"
chmod 600 "$SEED/user-data"
if [ -n "$RENDER_ONLY" ]; then echo "Seed geschrieben: $SEED"; rm -rf "$TMP"; exit 0; fi
rm -f "$WORK/seed.iso"
hdiutil makehybrid -quiet -iso -joliet -default-volume-name cidata -o "$WORK/seed.iso" "$SEED" || die "seed.iso nicht erstellbar"

# ----------------------------------------------------------------------------- 5. SSD auswählen
say "Externe Laufwerke"
diskutil list external physical || true
echo
read -r -p "Welches Laufwerk ist die SSD? (z. B. disk4): " DISK
DISK="${DISK#/dev/}"
[[ "$DISK" =~ ^disk[0-9]+$ ]] || die "Ungültige Angabe: $DISK"
diskutil info "/dev/$DISK" | grep -E "Device / Media Name|Disk Size|Protocol|Internal|Removable" || die "Laufwerk nicht gefunden"
if diskutil info "/dev/$DISK" | grep -qE "Internal: +Yes"; then die "/dev/$DISK ist ein internes Laufwerk – abgebrochen."; fi
echo
echo "ACHTUNG: Alle Daten auf /dev/$DISK werden gelöscht."
read -r -p "Zum Fortfahren LÖSCHEN eintippen: " OK
[ "$OK" = "LÖSCHEN" ] || [ "$OK" = "LOESCHEN" ] || die "abgebrochen"
diskutil unmountDisk force "/dev/$DISK" >/dev/null || true

# ----------------------------------------------------------------------------- 6. Installation in QEMU
say "Starte die unbeaufsichtigte Installation auf /dev/$DISK"
cp "$OVMF_VARS_TEMPLATE" "$WORK/ovmf-vars.fd"
if [ "$(uname -m)" = "x86_64" ]; then ACCEL=(-accel hvf -cpu host); else ACCEL=(-accel tcg,thread=multi -cpu max); fi
NCPU="$(sysctl -n hw.ncpu)"; [ "$NCPU" -gt 8 ] && NCPU=8
echo "Das dauert auf Apple Silicon 30–90 Minuten (x86 wird emuliert). Ausgabe der Installation folgt;"
echo "die VM schaltet sich am Ende selbst aus. Mac nicht in den Ruhezustand schicken (caffeinate läuft)."
echo "Administratorpasswort des Macs wird für den Zugriff auf die SSD gebraucht."
echo "Beenden im Notfall: Ctrl-A, dann X."
sudo caffeinate -i "$QEMU_BIN" \
  -machine q35 "${ACCEL[@]}" -smp "$NCPU" -m 6144 \
  -drive if=pflash,format=raw,readonly=on,file="$OVMF_CODE" \
  -drive if=pflash,format=raw,file="$WORK/ovmf-vars.fd" \
  -drive file="/dev/r$DISK",format=raw,if=virtio \
  -drive file="$ISO",media=cdrom,readonly=on \
  -drive file="$WORK/seed.iso",media=cdrom,readonly=on \
  -kernel "$KDIR/$KERNEL_PATH" -initrd "$KDIR/$INITRD_PATH" \
  -append "autoinstall console=ttyS0,115200n8 ---" \
  -netdev user,id=n0 -device virtio-net-pci,netdev=n0 \
  -nographic \
  || die "QEMU wurde mit Fehler beendet"

sync; diskutil list "/dev/$DISK" || true
say "Fertig."
cat <<EOT

  1. SSD vom Mac trennen (falls macOS meckert, dass sie nicht lesbar ist: "Ignorieren").
  2. SSD in den PV-Mapper-PC einbauen/anschließen, Netzwerkkabel stecken, einschalten.
     Falls er nicht von der SSD startet: im BIOS/UEFI (Entf oder F2) die SSD als erstes Boot-Laufwerk wählen.
  3. Am Mac im Browser:  http://$PC_HOSTNAME.local:8765
     Zuerst erscheint die Einrichtungsseite (10–30 Min.), danach PV-Mapper.
     SSH:  ssh pvmapper@$PC_HOSTNAME.local      Aktualisieren:  ssh … sudo pvmapper-update

EOT
rm -rf "$TMP"
