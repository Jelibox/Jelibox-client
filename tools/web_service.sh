#!/usr/bin/env bash
# Starts the Jelibox web server by itself when this computer boots. No sudo for the service itself.
#
#   bash tools/web_service.sh install|remove|start|stop|status [Jelibox folder]
#
#   install   writes a systemd USER unit (~/.config/systemd/user/jelibox-web.service), enables it and starts the
#             server now. Run again after an update: the server is restarted with the new code.
#   remove    stops the server, disables it and deletes the unit.
#   start / stop / status   do what they say.
#
# A user unit normally runs only while you are signed in. To have it start at boot without anybody signing in,
# "loginctl enable-linger" is turned on for your user (one sudo, only when your system does not allow it without).
# If you say no to it, or it fails, the server simply runs while you are signed in. The server's output goes to the
# journal:  journalctl --user -u jelibox-web
# Other devices on the network also need the firewall to allow the port, see the note printed by "install".
#
# Optional environment variables:
#   JELIBOX_SYSTEMD_DIR  write the unit here instead of ~/.config/systemd/user (tests)
#   JELIBOX_WEB_WAIT     seconds to wait for the server to answer after it was started (default 60)

action="${1:-status}"
root="${2:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
root="$(cd -- "$root" && pwd)"
unit_name="jelibox-web.service"
unit_dir="${JELIBOX_SYSTEMD_DIR:-${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user}"
unit="$unit_dir/$unit_name"
python="$root/jelibox/bin/python"
server="$root/utils/server.py"

say() { printf '[Jelibox] %s\n' "$*"; }

port() {
    local p
    p="$(sed -n 's/.*"web_port"[^0-9]*\([0-9][0-9]*\).*/\1/p' "$root/configs/_app.json" 2>/dev/null | head -n 1)"
    echo "${p:-8420}"
}

answers() { ( exec 3<>"/dev/tcp/127.0.0.1/$1" ) 2>/dev/null; }

have_user_systemd() {
    command -v systemctl >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1
}

wait_for_server() {
    local p="$1" i max
    max=$(( ${JELIBOX_WEB_WAIT:-60} * 2 ))
    for ((i = 0; i < max; i++)); do
        answers "$p" && return 0
        systemctl --user is-failed --quiet "$unit_name" 2>/dev/null && return 1
        sleep 0.5
    done
    return 1
}

show_addresses() {
    local p ip
    p="$(port)"
    say "The web version answers at:  http://localhost:$p"
    for ip in $(hostname -I 2>/dev/null); do
        case "$ip" in *:*|127.*) ;; *) say "From another device on the network:  http://$ip:$p" ;; esac
    done
}

firewall_note() {
    local p="$1"
    if command -v ufw >/dev/null 2>&1 && ufw status 2>/dev/null | grep -qi "status: active"; then
        say "The firewall (ufw) is on. For other devices run once:  sudo ufw allow $p/tcp"
    elif command -v firewall-cmd >/dev/null 2>&1 && firewall-cmd --state >/dev/null 2>&1; then
        say "The firewall (firewalld) is on. For other devices run once:  sudo firewall-cmd --permanent --add-port=$p/tcp && sudo firewall-cmd --reload"
    else
        say "If a firewall is on, it must allow port $p for other devices on the network."
    fi
}

enable_linger() {
    command -v loginctl >/dev/null 2>&1 || { say "loginctl not found - the server will run while you are signed in."; return 0; }
    local me="${USER:-$(id -un)}"
    [ "$(loginctl show-user "$me" -p Linger --value 2>/dev/null)" = "yes" ] && return 0
    loginctl enable-linger "$me" >/dev/null 2>&1 && { say "Start at boot is on (linger)."; return 0; }
    if command -v sudo >/dev/null 2>&1 && [ -r /dev/tty ] && [ -w /dev/tty ]; then
        say "Starting at boot, before anybody signs in, needs one administrator command: sudo loginctl enable-linger $me"
        if sudo loginctl enable-linger "$me" < /dev/tty; then
            say "Start at boot is on (linger)."
            return 0
        fi
    fi
    say "Start at boot is OFF: the server runs while you are signed in. To turn it on, as an administrator run:  loginctl enable-linger $me"
}

case "$action" in
    install)
        [ -f "$server" ] || { say "utils/server.py is missing in $root"; exit 1; }
        [ -x "$python" ] || { say "Jelibox's Python was not found: $python"; exit 1; }
        if ! have_user_systemd; then
            say "systemd user services are not available here (no systemctl, or no user session - over SSH try: loginctl enable-linger \$USER, then sign in again)."
            say "Jelibox web can still be started by hand:  \"$python\" -u \"$server\""
            exit 1
        fi
        mkdir -p "$unit_dir"
        esc() { printf '%s' "${1//%/%%}"; }
        cat > "$unit" <<EOF
[Unit]
Description=Jelibox web server
After=network-online.target

[Service]
WorkingDirectory=$(esc "$root")
ExecStart="$(esc "$python")" -u "$(esc "$server")"
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF
        systemctl --user daemon-reload
        systemctl --user enable "$unit_name" >/dev/null 2>&1 || { say "Could not enable $unit_name."; exit 1; }
        systemctl --user restart "$unit_name"           # an update must not keep running the old code
        say "Added the systemd user service $unit_name"
        enable_linger
        if wait_for_server "$(port)"; then
            say "The Jelibox web server is running."
            show_addresses
            firewall_note "$(port)"
            exit 0
        fi
        say "The server did not answer. Look at:  journalctl --user -u jelibox-web"
        exit 1
        ;;
    remove)
        if have_user_systemd; then
            systemctl --user disable --now "$unit_name" >/dev/null 2>&1 || true
        fi
        if [ -f "$unit" ]; then
            rm -f "$unit"
            have_user_systemd && systemctl --user daemon-reload || true
            say "Removed the systemd user service $unit_name (linger was left as it is)."
        fi
        exit 0
        ;;
    start)
        have_user_systemd && [ -f "$unit" ] || { say "Not installed. Run:  bash tools/web_service.sh install"; exit 1; }
        systemctl --user start "$unit_name"
        wait_for_server "$(port)" && { show_addresses; exit 0; }
        say "The server did not answer. Look at:  journalctl --user -u jelibox-web"
        exit 1
        ;;
    stop)
        have_user_systemd && systemctl --user stop "$unit_name" 2>/dev/null || true
        say "The web server was stopped."
        exit 0
        ;;
    status)
        p="$(port)"
        up=1
        answers "$p" && up=0
        if [ "$up" = 0 ]; then say "Web server: running on port $p"; else say "Web server: not running"; fi
        if [ -f "$unit" ] && have_user_systemd && systemctl --user is-enabled --quiet "$unit_name" 2>/dev/null; then
            say "Starts automatically: yes"
        else
            say "Starts automatically: no"
        fi
        exit "$up"
        ;;
    *)
        say "Usage: bash tools/web_service.sh install|remove|start|stop|status [Jelibox folder]"
        exit 2
        ;;
esac
