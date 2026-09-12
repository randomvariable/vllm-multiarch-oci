#!/bin/sh
export LD_LIBRARY_PATH=/opt/nccl/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=/opt/venv/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}
exec /opt/python/bin/python -m vllm "$@"
