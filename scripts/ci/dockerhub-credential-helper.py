#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Supply short-lived Docker Hub credentials to Bazel's downloader."""

import datetime
import json
import re
import subprocess
import sys
import urllib.parse


def main():
    uri = urllib.parse.urlsplit(json.load(sys.stdin)["uri"])
    path = re.fullmatch(r"/v2/(.+)/(?:blobs|manifests)/[^/]+", uri.path)
    if uri.scheme != "https" or uri.netloc != "index.docker.io" or path is None:
        raise ValueError("expected a Docker Hub blob or manifest URL")

    # crane uses the lane's existing DOCKER_CONFIG, including anonymous pulls.
    started = datetime.datetime.now(datetime.UTC)
    result = subprocess.run(
        ["crane", "auth", "token", "index.docker.io/" + path[1]],
        check=True,
        capture_output=True,
        text=True,
    )
    credentials = json.loads(result.stdout)
    # Expire before the registry does, including time spent acquiring the token.
    expires = started + datetime.timedelta(seconds=credentials["expires_in"] - 30)
    json.dump(
        {
            "headers": {"Authorization": ["Bearer " + credentials["token"]]},
            "expires": expires.isoformat(timespec="seconds"),
        },
        sys.stdout,
    )


if __name__ == "__main__":
    main()
