#!/bin/sh
set -eu

tar=$1
archive=$2
output=$3
mkdir -p "$output"
"$tar" -xf "$archive" -C "$output"
# qemu-user-binfmt depends on an Ubuntu Python package that leaves this
# absolute link unresolved. The ARM Python launcher uses only qemu itself.
rm -rf "$output/usr/lib/python3.14"
