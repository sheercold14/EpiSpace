#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_ROOT="$ROOT/.cache/sysdeps"
SYSROOT="$ROOT/.data/sysroot"
packages=(libglu1-mesa libopengl0)

command -v apt >/dev/null || {
  echo "apt is required to fetch the Ubuntu headless runtime libraries." >&2
  exit 2
}
command -v dpkg-deb >/dev/null || {
  echo "dpkg-deb is required to unpack the headless runtime libraries." >&2
  exit 2
}

mkdir -p "$CACHE_ROOT" "$SYSROOT"
for package in "${packages[@]}"; do
  package_cache="$CACHE_ROOT/$package"
  mkdir -p "$package_cache"
  if ! compgen -G "$package_cache/${package}_*.deb" >/dev/null; then
    (
      cd "$package_cache"
      apt download "$package"
    )
  fi
  for archive in "$package_cache"/${package}_*.deb; do
    dpkg-deb -x "$archive" "$SYSROOT"
  done
done

lib_dir="$SYSROOT/usr/lib/x86_64-linux-gnu"
[[ -e "$lib_dir/libGLU.so.1" && -e "$lib_dir/libOpenGL.so.0" ]] || {
  echo "Headless runtime extraction did not produce libGLU and libOpenGL." >&2
  exit 2
}

echo "Headless runtime installed at: $SYSROOT"
