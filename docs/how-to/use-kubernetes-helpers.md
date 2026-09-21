# Use the Kubernetes Image Helpers

The image includes `/opt/venv/bin/vllm-image`. These commands replace inline shell and Python helpers in Kubernetes manifests. They do not change the `/opt/venv/bin/vllm` serving command.

## Synchronize a Model Snapshot

Use `model-sync` in an init container that mounts persistent model storage. Supply a full lowercase 40-character Hugging Face commit OID. Branch and tag names are not accepted.

```yaml
initContainers:
  - name: model-sync
    image: ghcr.io/randomvariable/vllm-b12x-multi@sha256:<digest>
    command: [/opt/venv/bin/vllm-image]
    args:
      - model-sync
      - --repo
      - example/model
      - --revision
      - 0123456789abcdef0123456789abcdef01234567
      - --storage-root
      - /var/lib/models
      - --publish
      - /models/current
      - --min-free-gib
      - "200"
      - --workers
      - "8"
      - --ignore
      - examples/*
      - --require
      - chat_template.jinja
    env:
      - name: HF_TOKEN
        valueFrom:
          secretKeyRef: {name: huggingface-token, key: token}
    volumeMounts:
      - {name: model-store, mountPath: /var/lib/models}
      - {name: published-model, mountPath: /models}
```

The command sets `HF_HOME`, `HF_HUB_CACHE`, and `HF_XET_CACHE` below `--storage-root` unless the deployment already supplies them. A deployment that keeps the Xet chunk cache on a scratch volume therefore keeps that location, and the model store does not absorb the cache. It validates `config.json` and all shards named by each safetensors index. `--require` names additional snapshot-relative files that must exist and be non-empty, for assets the engine loads after the weights. It then atomically replaces `--publish` with a symlink to the validated snapshot. A revision-specific marker makes a valid completed download idempotent. Partial or stale snapshots do not receive a completion marker.

## Wait for the Leader Rendezvous

Run this command before the worker starts vLLM. Rank zero exits immediately. Other ranks resolve the leader and wait for the TCP port.

```yaml
initContainers:
  - name: rendezvous-wait
    image: ghcr.io/randomvariable/vllm-b12x-multi@sha256:<digest>
    command: [/opt/venv/bin/vllm-image]
    args:
      - rendezvous-wait
      - --rank-env
      - LWS_WORKER_INDEX
      - --leader-env
      - LWS_LEADER_ADDRESS
      - --port
      - "25000"
      - --dns-timeout
      - 20m
      - --connect-timeout
      - 5m
```

Use `--rank` or `--leader` to override environment-derived values. DNS and TCP failures use different exit codes and messages.

## Configure Startup and Readiness Probes

The leader checks its local API. A worker checks the leader API and exactly one live direct engine child of PID 1. The child must have the exact Linux process name `VLLM::Worker_TP` or `VLLM::EngineCor`.

```yaml
startupProbe:
  exec:
    command:
      - /opt/venv/bin/vllm-image
      - health
      - --phase
      - startup
      - --rank-env
      - LWS_WORKER_INDEX
      - --local-url
      - http://127.0.0.1:8888/v1/models
      - --leader-host-env
      - LWS_LEADER_ADDRESS
      - --leader-port
      - "8888"
      - --leader-path
      - /v1/models
      - --engine-parent-pid
      - "1"
      - --timeout
      - 5s
readinessProbe:
  exec:
    command:
      - /opt/venv/bin/vllm-image
      - health
      - --phase
      - readiness
      - --rank-env
      - LWS_WORKER_INDEX
      - --local-url
      - http://127.0.0.1:8888/v1/models
      - --leader-host-env
      - LWS_LEADER_ADDRESS
      - --leader-port
      - "8888"
      - --leader-path
      - /v1/models
      - --engine-parent-pid
      - "1"
      - --timeout
      - 5s
```

Use `--leader-url` to override the environment-derived leader URL. Use `--engine-pid` to override child discovery when the launcher provides a stable PID.

Do not configure an exec or HTTP liveness probe for this image. vLLM is PID 1. Kubernetes already observes terminal process death. A liveness request can time out during a long GPU operation and restart a healthy distributed group.
