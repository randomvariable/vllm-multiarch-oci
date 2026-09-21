# Run the Image as a Non-Root User

The image names one non-root account, so a pod that pins its own `runAsUser` has an account that resolves.

| Account | uid | gid | Home | Shell |
| --- | --- | --- | --- | --- |
| `vllm` | 10001 | 10001 | `/home/vllm` | `/bin/bash` |

The image leaves `USER` unset, so the default user stays root, and `WORKDIR` stays `/root`. Root is the only account that can use that working directory.

## Pin the Account in a Pod

```yaml
spec:
  securityContext:
    runAsUser: 10001
    runAsGroup: 10001
  containers:
    - name: server
      image: ghcr.io/randomvariable/vllm-b12x-multi@sha256:<digest>
      env:
        - {name: HOME, value: /cache/vllm-home}
        - {name: FLASHINFER_WORKSPACE_BASE, value: /cache/flashinfer}
        - {name: B12X_COMPILE_CACHE_DIR, value: /cache/b12x-compile}
      volumeMounts:
        - {name: cache, mountPath: /cache}
  volumes:
    - name: cache
      hostPath: {path: /var/lib/vllm-cache, type: DirectoryOrCreate}
```

Set `HOME`, `FLASHINFER_WORKSPACE_BASE` and `B12X_COMPILE_CACHE_DIR` explicitly. Treat all three as required for a pod that sets its own `runAsUser`, and do not rely on `~` being resolved for you.

`B12X_COMPILE_CACHE_DIR` is the same class of defect as the other two. B12X resolves its generated-kernel cache as `B12X_COMPILE_CACHE_DIR`, then `$XDG_CACHE_HOME/b12x/compile`, then `~/.cache/b12x/compile`, and that last step is why the location can end up inherited from its environment instead of chosen. Measured by the Qwen deployment on this image: engine initialisation took 172 s cold against 61 s warm, so losing the cache costs about two minutes on every start, silently. The image cannot decide this for you, because the correct path is a mounted volume rather than anything inside the container.

One variable decides more than the compiled kernels. The preparation selection cache takes its root from the same directory (`b12x/preparation/session.py:351` calls `_cute_compile_cache_dir()`), so moving this variable also discards the tuning selections and costs one full autotune at the next start. Its identity covers the device name, model, dtype, KV-cache dtype, the parallelism sizes, the speculative configuration and `B12X_TUNING_CACHE_VERSION` (`b12x/preparation/_cache.py:25`, namespace assembled in `vllm/model_executor/warmup/b12x_prepare.py:491`); the image version is not part of it, so a pin bump alone leaves it valid.

## Point HOME at Durable Storage

Do not leave `HOME` pointing into the image, and do not treat the named account as a reason to drop the variable.

The account makes the lookup resolve. `/home/vllm` is still a path inside the container filesystem, so anything a component derives from `~` is lost when the pod restarts, and every byte written there counts against the pod's ephemeral-storage limit. Triton, Inductor, FlashInfer, and Hugging Face caches all derive their workspace from `HOME`, so a home inside the image turns a durable cache into a per-pod one that can also evict the pod.

Point `HOME` at a mounted volume, as in the example above.

## What a Missing Account Breaks

A uid without a passwd entry has no home directory. `expanduser("~")` then returns `/`, and the first import that creates a workspace below the home directory fails. FlashInfer JIT reports this as a failed `makedirs` while `vllm.compilation.backends` is imported, which aborts the engine before it starts.

The named account removes that failure. The two variables decide which paths a library derives from `~`, and keep those paths on storage you control.

## Numeric Ownership in Init Containers

`install -d -o <uid>` and `install -d -g <gid>` do not accept numeric owners or groups in this image. The base supplies the uutils `install`, which resolves `-o` and `-g` through `getpwnam` and `getgrnam`, and reports `invalid user: '10001'`. Mode-only use, such as `install -m 0644 <source> <target>`, is unaffected.

Use `mkdir` and `chown` instead:

```sh
mkdir -p /var/lib/cache
chown 10001:10001 /var/lib/cache
```

`chown` accepts numeric ids, so this idiom works on this image and on any other Ubuntu 26.04 base. Do not repoint `/usr/bin/install` at GNU coreutils to recover the `install -o` form: the package that provides it installs prefixed binaries (`/usr/bin/gnuinstall`), and a manifest that depends on a shadowed `/usr/bin/install` fails on every base that does not shadow it.
