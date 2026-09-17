# Mooncake Transfer Engine in the VLLMB12X Image

The VLLMB12X image adds the CUDA 13 Mooncake Transfer Engine to the base vLLM runtime. The addition makes the Python package and its command-line tools available for a deployment that selects vLLM's `MooncakeStoreConnector` and TCP transport. It does not enable a connector, start Mooncake services, create cache storage, or change vLLM's default key-value cache behavior.

## Added Runtime Surface

| Item | Version or path | Purpose |
| --- | --- | --- |
| `mooncake-transfer-engine-cuda13` | `0.3.13.post1` | CUDA 13 transfer engine, its `mooncake` Python module, and native binaries. |
| `paramiko` | `5.0.0` | Mooncake's SSD registration and SPDK target modules. |
| Mooncake commands | `/opt/venv/bin/{mooncake_master,mooncake_client,transfer_engine_bench,mooncake_http_metadata_server,mc_store_rest_server,transfer_engine_topology_dump}` | Package-provided administrative, benchmark, and service commands. |

The image already supplies Mooncake's declared base Python dependencies: `aiohttp`, `requests`, and `msgpack`.

Mooncake publishes the CUDA 13 distribution at [PyPI](https://pypi.org/project/mooncake-transfer-engine-cuda13/). The upstream project owns the package and its transport behavior at [kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake). This repository only pins the package, assembles it into the image, and verifies its installed surface.

## Assembly Behavior

The runtime layer uses `pip install --target` for reproducible image assembly. That pip mode installs package code but omits wheel-provided console scripts. `bazel/venv_layer_action.py` now materializes each installed distribution's `console_scripts` metadata into `/opt/venv/bin`; this applies to every runtime wheel, not only Mooncake. The wrappers invoke the image's hermetic Python launcher and preserve command-line arguments.

The image excludes the PyPI `instanttensor` wheel. The profile instead installs its pinned source-built `instanttensor` `0.1.9` wheel. This avoids placing upstream vLLM's unconstrained `0.2.0` wheel beside the profile-selected build.

## Release Contract

A release lists this as an image addition over base vLLM. It does not list it in the upstream-change ledger because Mooncake is not a source change in the pinned vLLM or B12X revisions.

The Docker-backed image contract checks the exact Mooncake distribution version, the `mooncake` Python module, all six declared console entry points, and their wrappers. Build and test with:

```bash
bazel test --config=remote-aarch64 --config=remote-gpu \
  //tests/image:vllmb12x_contract --test_output=errors
```

A passing contract proves that the ARM64 image contains this runtime surface. It does not prove a Mooncake service deployment, TCP reachability, or cross-instance prefix-cache transfers. Validate those in the target Kubernetes deployment with its selected connector configuration and peers.
