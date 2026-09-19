#!/bin/sh
# `tar` is a declared Bazel tool input. Extract every package payload without
# consulting a worker's tar, compiler, headers, or libraries.
set -eu

tar=$1
output=$2
shift 2
mkdir -p "$output"
for archive in "$@"; do
    "$tar" -xzf "$archive" -C "$output"
done
 ln -s usr/lib "$output/lib"
mkdir -p "$output/lib64"
loader=$(find "$output/usr/lib" -name 'ld-linux-*.so.*' -type f -print -quit)
if [ -n "$loader" ]; then
    relative=${loader#"$output/usr/lib/"}
    ln -s "../usr/lib/$relative" "$output/lib64/$(basename "$loader")"
fi

# Debian x86 linker scripts name their shared objects with absolute /usr paths.
# The sandbox has no host /usr tree, so make those paths sysroot-relative.
for script in "$output"/usr/lib/x86_64-linux-gnu/*.so; do
    [ -f "$script" ] || continue
    sed 's#\(^\|[[:space:](]\)/usr/lib/#\1=/usr/lib/#g' "$script" > "$script.rewritten"
    mv "$script.rewritten" "$script"
done
rm -rf "$output/usr/share/man"
