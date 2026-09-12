#!/bin/sh
export PYTHONPATH=/opt/venv/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}
exec /opt/python/bin/python "$@"
