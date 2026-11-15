#!/bin/bash
# Writes /etc/netplan/91-pvmapper-wifi.yaml for every Wi-Fi interface (wl*) of this PC from /etc/pvmapper/wifi
# (SSID=…, PASSWORD=…). netplan's wpa_supplicant integration needs the real interface name, which differs per board.
set -u
CONF=/etc/pvmapper/wifi
OUT=/etc/netplan/91-pvmapper-wifi.yaml
[ -f "$CONF" ] || exit 0
. "$CONF"
[ -n "${SSID:-}" ] || { rm -f "$OUT"; exit 0; }
q() { printf '"%s"' "$(printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g')"; }
IFACES=$(ls /sys/class/net | grep '^wl' || true)
[ -n "$IFACES" ] || exit 0
{
  echo "network:"
  echo "  version: 2"
  echo "  renderer: networkd"
  echo "  wifis:"
  for i in $IFACES; do
    echo "    $i:"
    echo "      dhcp4: true"
    echo "      optional: true"
    echo "      access-points:"
    echo "        $(q "$SSID"):"
    echo "          password: $(q "${PASSWORD:-}")"
  done
} > "$OUT.tmp"
chmod 600 "$OUT.tmp"
if ! cmp -s "$OUT.tmp" "$OUT"; then mv "$OUT.tmp" "$OUT"; netplan generate; else rm -f "$OUT.tmp"; fi
