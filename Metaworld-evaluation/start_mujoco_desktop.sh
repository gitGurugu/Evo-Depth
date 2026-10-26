#!/usr/bin/env bash
# 四块虚拟桌面，每路 MuJoCo 查看器一个网页。Xvfb 已在就不动。
# x11vnc 每次都重开，转发整块桌面。不要绑窗口 id，窗口关掉时转发要留着。
# easy :1 -> http://192.168.2.1:6080/vnc.html?resize=scale&autoconnect=1
# medium :2 -> http://192.168.2.1:6081/vnc.html?resize=scale&autoconnect=1
# hard :3 -> http://192.168.2.1:6082/vnc.html?resize=scale&autoconnect=1
# very_hard :4 -> http://192.168.2.1:6083/vnc.html?resize=scale&autoconnect=1
set -euo pipefail

ROOT=$HOME/opt/vnc
export LD_LIBRARY_PATH="$ROOT/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
source "$HOME/Evo-Depth/Metaworld-evaluation/.venv-metaworld/bin/activate"

fill_window() {
  DISPLAY=:$1 python - << 'PY'
import ctypes, os
lib = ctypes.CDLL("libX11.so.6")
lib.XOpenDisplay.restype = ctypes.c_void_p
lib.XDefaultScreen.argtypes = [ctypes.c_void_p]
lib.XDisplayWidth.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.XDisplayWidth.restype = ctypes.c_int
lib.XDisplayHeight.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.XDisplayHeight.restype = ctypes.c_int
lib.XRootWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.XRootWindow.restype = ctypes.c_ulong
P = ctypes.POINTER
lib.XQueryTree.argtypes = [ctypes.c_void_p, ctypes.c_ulong, P(ctypes.c_ulong), P(ctypes.c_ulong), P(P(ctypes.c_ulong)), P(ctypes.c_uint)]
lib.XFetchName.argtypes = [ctypes.c_void_p, ctypes.c_ulong, P(ctypes.c_char_p)]
lib.XMoveResizeWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint]
lib.XFree.argtypes = [ctypes.c_void_p]
lib.XFlush.argtypes = [ctypes.c_void_p]
lib.XCloseDisplay.argtypes = [ctypes.c_void_p]
d = lib.XOpenDisplay(None)
if not d:
    raise SystemExit(0)
s = lib.XDefaultScreen(d)
w, h = lib.XDisplayWidth(d, s), lib.XDisplayHeight(d, s)
root = lib.XRootWindow(d, s)
rr, par = ctypes.c_ulong(), ctypes.c_ulong()
ch, n = P(ctypes.c_ulong)(), ctypes.c_uint()
lib.XQueryTree(d, root, ctypes.byref(rr), ctypes.byref(par), ctypes.byref(ch), ctypes.byref(n))
for i in range(n.value):
    name = ctypes.c_char_p()
    if lib.XFetchName(d, ch[i], ctypes.byref(name)) and name.value and b"MuJoCo" in name.value:
        lib.XMoveResizeWindow(d, ch[i], 0, 0, w, h)
if n.value:
    lib.XFree(ch)
lib.XFlush(d)
lib.XCloseDisplay(d)
PY
}

start_one() {
  local n=$1
  local vnc=$((5900 + n))
  local web=$((6079 + n))
  if [[ ! -S /tmp/.X11-unix/X$n ]]; then
    "$ROOT/usr/bin/Xvfb" ":$n" -screen 0 1280x800x24 -ac +extension GLX +render -noreset \
      > "$ROOT/xvfb-$n.log" 2>&1 &
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      [[ -S /tmp/.X11-unix/X$n ]] && break
      sleep 0.2
    done
  fi
  fill_window "$n" || true
  local -a olds=()
  local old
  while read -r old; do
    [[ -n "$old" ]] && olds+=("$old")
  done < <(ps -u "$(id -u)" -o pid=,args= | awk -v n="$n" '$0 ~ "x11vnc" && $0 ~ ("-display :" n " ") {print $1}')
  if ((${#olds[@]})); then
    kill -9 "${olds[@]}" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      ss -ltn | grep -q ":$vnc " || break
      sleep 0.1
    done
  fi
  "$ROOT/usr/bin/x11vnc" -display ":$n" -forever -shared -rfbport "$vnc" \
    -passwdfile "$HOME/.vnc/x11vnc.pass" \
    -noxdamage -nonap -wait 16 -defer 10 \
    -bg -o "$ROOT/x11vnc-$n.log" >/dev/null || true
  sleep 0.3
  ss -ltn | grep -q ":$vnc " || { echo "x11vnc :$n failed" >&2; tail -n 20 "$ROOT/x11vnc-$n.log" >&2; exit 1; }
  if ! ss -ltn | grep -q ":$web "; then
    nohup websockify --web "$ROOT/usr/share/novnc" "$web" "localhost:$vnc" \
      > "$ROOT/websockify-$n.log" 2>&1 &
    sleep 0.3
    ss -ltn | grep -q ":$web " || { echo "websockify :$n failed" >&2; tail -n 20 "$ROOT/websockify-$n.log" >&2; exit 1; }
  fi
  echo ":$n  http://192.168.2.1:${web}/vnc.html?resize=scale&autoconnect=1"
}

for n in 1 2 3 4; do
  start_one "$n"
done

# 新窗口出来后拉满。已有看守就不动。
pidfile=$ROOT/resize.pid
if [[ ! -f $pidfile ]] || ! kill -0 "$(cat "$pidfile")" 2>/dev/null; then
  (
    while true; do
      for n in 1 2 3 4; do
        [[ -S /tmp/.X11-unix/X$n ]] || continue
        DISPLAY=:$n xwininfo -root -tree 2>/dev/null | grep -q 'MuJoCo : MuJoCo Model".*1280x800+0+0' \
          || fill_window "$n" || true
      done
      sleep 1
    done
  ) >/dev/null 2>&1 &
  echo $! > "$pidfile"
fi
