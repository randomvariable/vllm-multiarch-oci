"""In-image launcher: resolve a model policy, orchestrate one container, exec.

This package is the single home for the launcher's error type and process exit
codes. Every module here must stay importable with the standard library and
PyYAML only, because ``resolver`` also runs in the browser under Pyodide.
"""

from __future__ import annotations

EXIT_CONFIG = 2
EXIT_DNS = 10
EXIT_PORT = 11
EXIT_HTTP = 20
EXIT_ENGINE = 21
EXIT_MODEL = 30
EXIT_CACHE = 40


class ConfigError(Exception):
    """A configuration or health precondition that cannot be satisfied.

    ``code`` is the process exit status the command line reports for this
    failure, so a caller can tell a bad setting apart from a missing model or
    an unhealthy engine without matching on the message.
    """

    def __init__(self, message: str, code: int = EXIT_CONFIG) -> None:
        super().__init__(message)
        self.code = code
