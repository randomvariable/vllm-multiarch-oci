#!/bin/sh
set -eu
if [ "$(uname -m)" = aarch64 ]; then
    export PATH="$(dirname "$0"):$PATH"
    exec "{python}" "$@"
fi
export PATH="$(dirname "$0"):$PATH"
exec "{qemu_root}/usr/bin/qemu-aarch64" -L "{sysroot}" "{python}" "$@"
