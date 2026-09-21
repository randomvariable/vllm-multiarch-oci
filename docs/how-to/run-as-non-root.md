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
        - {name: HOME, value: /home/vllm}
        - {name: FLASHINFER_WORKSPACE_BASE, value: /home/vllm/flashinfer}
```

Set `HOME` and `FLASHINFER_WORKSPACE_BASE` explicitly. Treat both as required for a pod that sets its own `runAsUser`, and do not rely on `~` being resolved for you.

## Why Both Variables Are Required

A uid without a passwd entry has no home directory. `expanduser("~")` then returns `/`, and the first import that creates a workspace below the home directory fails. FlashInfer JIT reports this as a failed `makedirs` while `vllm.compilation.backends` is imported, which aborts the engine before it starts.

The named account fixes the lookup. The two variables fix the paths a library derives from it, and they keep the workspace on storage you control rather than in the image layer.

## Numeric Ownership in Init Containers

`install -d -o <uid>` and `install -d -g <gid>` do not accept numeric owners or groups in this image. The base supplies the uutils `install`, which resolves `-o` and `-g` through `getpwnam` and `getgrnam`, and reports `invalid user: '10001'`. Mode-only use, such as `install -m 0644 <source> <target>`, is unaffected.

Use `mkdir` and `chown` instead:

```sh
mkdir -p /var/lib/cache
chown 10001:10001 /var/lib/cache
```

`chown` accepts numeric ids, so this idiom works on this image and on any other Ubuntu 26.04 base. Do not repoint `/usr/bin/install` at GNU coreutils to recover the `install -o` form: the package that provides it installs prefixed binaries (`/usr/bin/gnuinstall`), and a manifest that depends on a shadowed `/usr/bin/install` fails on every base that does not shadow it.
