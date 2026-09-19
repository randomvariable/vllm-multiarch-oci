# Local builds never load private Bazel rc files.
set positional-arguments

platform := "//platforms:spark_arm64_sm121"
cache_root := env_var_or_default("VLLMB12X_CACHE_ROOT", "")

# List available tasks.
default:
    @just --list

# Run Bazel without host or private rc files; pass command-specific flags explicitly.
bazel +args:
    bazel --ignore_all_rc_files "$@"

# Analyse the local ARM64 image graph without compiling.
analyze:
    @just build --nobuild

# Compile and link a C++ binary with only the locked GCC package closure.
toolchain-check:
    python3 -m unittest bazel/hermetic_cc_toolchain_test.py

# Build one platform leaf locally; extra arguments are forwarded to Bazel.
build *args:
    bazel --ignore_all_rc_files build //image:vllmb12x_image --platforms={{platform}} --extra_execution_platforms=//platforms:local_x86_64,{{platform}} --spawn_strategy=local --disk_cache=.bazel-cache --incompatible_strict_action_env --//platforms:vllmb12x_cache_root={{cache_root}} "$@"

# Build the two-platform OCI index locally; extra arguments are forwarded to Bazel.
build-multiarch *args:
    bazel --ignore_all_rc_files build //image:vllmb12x --extra_execution_platforms=//platforms:local_x86_64,//platforms:spark_arm64_sm121,//platforms:blackwell_x86_64_sm120 --extra_toolchains=//platforms:hermetic_linux_aarch64_cc_toolchain,//platforms:hermetic_linux_x86_64_cc_toolchain --spawn_strategy=local --disk_cache=.bazel-cache --incompatible_strict_action_env --//platforms:vllmb12x_cache_root={{cache_root}} "$@"

# Run the local Docker-backed image contract.
test *args:
    bazel --ignore_all_rc_files test //tests/image:vllmb12x_contract --platforms={{platform}} --extra_execution_platforms=//platforms:local_x86_64,{{platform}} --spawn_strategy=local --disk_cache=.bazel-cache --incompatible_strict_action_env --//platforms:vllmb12x_cache_root={{cache_root}} --test_output=errors "$@"

# Load the image into the local container daemon without starting it.
load *args:
    bazel --ignore_all_rc_files run //image:vllmb12x_image_load --platforms={{platform}} --extra_execution_platforms=//platforms:local_x86_64,{{platform}} --spawn_strategy=local --disk_cache=.bazel-cache --incompatible_strict_action_env --//platforms:vllmb12x_cache_root={{cache_root}} "$@"

# Resolve the moving upstream branch into immutable source pins.
refresh-vllmb12x:
    python3 scripts/refresh-vllmb12x.py

# Serve the recipes site locally with live reload; extra arguments reach astro dev.
serve *args:
    cd recipes-site && pnpm install --frozen-lockfile && pnpm run dev "$@"
