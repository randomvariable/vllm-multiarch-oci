#!/usr/bin/env python3
"""Write the resolved source lock as an OCI layer.

The build input is a moving upstream branch resolved to one commit before the
build starts. Carrying the resolved manifest, the generated build version and
every source identity, including applied patch hashes, makes a published digest
reconstructable from the image alone.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from action_lib import work_root, write_tar

LOCK_ROOT = "opt/vllmb12x"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--build-version", required=True)
    parser.add_argument("--source-identity", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    execroot = Path.cwd()
    root = work_root(args.work_dir) / "root"
    lock_dir = root / LOCK_ROOT
    identity_dir = lock_dir / "source-identity"
    identity_dir.mkdir(parents=True)

    index = {}
    for entry in sorted(args.source_identity):
        name, separator, path = entry.partition("=")
        if not separator:
            raise SystemExit(f"--source-identity needs <name>=<path>: {entry}")
        payload = (execroot / path).read_bytes()
        (identity_dir / f"{name}.json").write_bytes(payload)
        index[name] = {
            "path": f"source-identity/{name}.json",
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    manifest = json.loads((execroot / args.manifest).read_text())
    if set(manifest["sources"]) != set(index):
        raise SystemExit(
            "source identities do not cover the manifest: "
            f"{sorted(set(manifest['sources']) ^ set(index))}"
        )
    lock = {
        "build_version": args.build_version,
        "profile": manifest,
        "source_identity": index,
    }
    (lock_dir / "sources.lock.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n"
    )
    write_tar(execroot / args.output, root)


if __name__ == "__main__":
    main()
