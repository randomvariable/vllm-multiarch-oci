#!/usr/bin/env python3
"""Add named non-root accounts to the image's passwd and group files.

A pod that pins its own ``runAsUser`` has no passwd entry for that uid, so
``expanduser("~")`` resolves to ``/`` and any import-time ``makedirs`` under the
home directory fails. Ubuntu resolves accounts through ``/etc/passwd``, so the
account has to exist there.

OCI layers replace whole files, so this layer carries complete ``/etc/passwd``
and ``/etc/group`` files. They are read from the pinned base image itself
instead of being vendored here, because a vendored copy would silently drift
from the base the next time the base digest moves. The newest layer that
defines a path is the one the runtime applies last, so the scan walks the base
layers newest-first and takes the first definition of each path.
"""

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from action_lib import work_root

# Paths this layer owns in the image filesystem.
PASSWD = "etc/passwd"
GROUP = "etc/group"
# The base's own modes. The accounts files are world readable and root owned.
ACCOUNTS_MODE = 0o644
HOME_MODE = 0o755
MAX_UID = 60000


def base_layers(layout: Path) -> list[dict]:
    """Return the base image's layer descriptors, oldest first."""
    index = json.loads((layout / "index.json").read_text())
    manifests = index.get("manifests") or []
    if len(manifests) != 1:
        raise SystemExit(
            f"{layout} must hold a single-platform base, found {len(manifests)} manifests"
        )
    digest = manifests[0]["digest"].split(":", 1)[1]
    manifest = json.loads((layout / "blobs" / "sha256" / digest).read_text())
    return manifest["layers"]


def read_applied_files(
    layout: Path, layers: list[dict], wanted: set[str]
) -> dict[str, str]:
    """Return the content each path has after every base layer is applied."""
    resolved: dict[str, str] = {}
    for layer in reversed(layers):
        blob = layout / "blobs" / "sha256" / layer["digest"].split(":", 1)[1]
        with tarfile.open(blob, "r|*") as archive:
            for member in archive:
                name = member.name.removeprefix("./")
                if name in wanted and name not in resolved:
                    handle = archive.extractfile(member)
                    if handle is None:
                        raise SystemExit(f"{name} is not a regular file in {blob}")
                    resolved[name] = handle.read().decode()
        if len(resolved) == len(wanted):
            break
    missing = sorted(wanted - set(resolved))
    if missing:
        raise SystemExit(f"base image does not define {missing}")
    return resolved


def parse_account(spec: str) -> dict:
    fields = spec.split(":")
    if len(fields) != 6:
        raise SystemExit(f"--account takes name:uid:gid:gecos:home:shell, got {spec!r}")
    name, uid, gid, gecos, home, shell = fields
    if not name or ":" in name or name.startswith("-"):
        raise SystemExit(f"unusable account name {name!r}")
    uid_value, gid_value = int(uid), int(gid)
    if not 0 < uid_value <= MAX_UID or not 0 < gid_value <= MAX_UID:
        raise SystemExit(f"account {name} needs a uid and gid in 1..{MAX_UID}")
    if not home.startswith("/") or home == "/":
        raise SystemExit(
            f"account {name} needs an absolute home directory, got {home!r}"
        )
    return {
        "name": name,
        "uid": uid_value,
        "gid": gid_value,
        "gecos": gecos,
        "home": home,
        "shell": shell,
    }


def parse_group(spec: str) -> dict:
    fields = spec.split(":")
    if len(fields) != 2:
        raise SystemExit(f"--group takes name:gid, got {spec!r}")
    name, gid = fields
    if not name or name.startswith("-"):
        raise SystemExit(f"unusable group name {name!r}")
    return {"name": name, "gid": int(gid)}


def require_free(
    accounts: list[dict], groups: list[dict], passwd: str, group: str
) -> None:
    """Fail rather than shadow or duplicate an account the base already defines."""
    taken_uids = {
        line.split(":")[2] for line in passwd.splitlines() if line.count(":") >= 6
    }
    taken_names = {
        line.split(":")[0] for line in passwd.splitlines() if line.count(":") >= 6
    }
    taken_gids = {
        line.split(":")[2] for line in group.splitlines() if line.count(":") >= 3
    }
    for account in accounts:
        if account["name"] in taken_names:
            raise SystemExit(
                f"account {account['name']} already exists in the base image"
            )
        if str(account["uid"]) in taken_uids:
            raise SystemExit(f"uid {account['uid']} already exists in the base image")
    for entry in groups:
        if str(entry["gid"]) in taken_gids:
            raise SystemExit(f"gid {entry['gid']} already exists in the base image")


def write_layer(
    archive_path: Path, entries: list[tuple[str, bytes | None, int, int, int]]
) -> None:
    """Write a deterministic layer that records ownership for each entry.

    ``action_lib.write_tar`` normalises every entry to root, which cannot express
    an account's own home directory, so this writes the tar directly.
    """
    with tarfile.open(archive_path, "w", format=tarfile.PAX_FORMAT) as output:
        for arcname, payload, mode, uid, gid in sorted(entries):
            info = tarfile.TarInfo(arcname)
            info.mode = mode
            info.uid = uid
            info.gid = gid
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            if payload is None:
                info.type = tarfile.DIRTYPE
                output.addfile(info)
            else:
                info.size = len(payload)
                output.addfile(info, io.BytesIO(payload))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="pulled base image layout")
    parser.add_argument("--account", action="append", default=[])
    parser.add_argument("--group", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    layout = work_root(args.base)
    accounts = sorted(
        (parse_account(spec) for spec in args.account), key=lambda a: a["uid"]
    )
    groups = sorted((parse_group(spec) for spec in args.group), key=lambda g: g["gid"])
    if not accounts:
        raise SystemExit("no accounts requested")

    applied = read_applied_files(layout, base_layers(layout), {PASSWD, GROUP})
    passwd, group = applied[PASSWD], applied[GROUP]
    require_free(accounts, groups, passwd, group)

    passwd_lines = passwd.splitlines()
    group_lines = group.splitlines()
    entries: list[tuple[str, bytes | None, int, int, int]] = []
    for account in accounts:
        passwd_lines.append(
            ":".join(
                [
                    account["name"],
                    "x",
                    str(account["uid"]),
                    str(account["gid"]),
                    account["gecos"],
                    account["home"],
                    account["shell"],
                ]
            )
        )
        entries.append(
            (
                account["home"].lstrip("/"),
                None,
                HOME_MODE,
                account["uid"],
                account["gid"],
            )
        )
    for entry in groups:
        group_lines.append(":".join([entry["name"], "x", str(entry["gid"]), ""]))

    entries.append(
        (PASSWD, ("\n".join(passwd_lines) + "\n").encode(), ACCOUNTS_MODE, 0, 0)
    )
    entries.append(
        (GROUP, ("\n".join(group_lines) + "\n").encode(), ACCOUNTS_MODE, 0, 0)
    )

    output = Path.cwd() / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    write_layer(output, entries)

    for account in accounts:
        print(
            f"account {account['name']} uid={account['uid']} home={account['home']}",
            file=sys.stderr,
        )
    print(f"wrote {output} with {len(accounts)} account(s)", file=sys.stderr)


if __name__ == "__main__":
    main()
