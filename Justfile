# Local builds never load private Bazel rc files.
set positional-arguments

platform := "//platforms:spark_arm64_sm121"

# List available tasks.
default:
    @just --list

# Run Bazel without host or private rc files; pass command-specific flags explicitly.
bazel +args:
    bazel --ignore_all_rc_files "$@"

# Analyse the local ARM64 image graph without compiling.
analyze:
    @just build --nobuild

# Build locally; extra arguments are forwarded to Bazel.
build *args:
    bazel --ignore_all_rc_files build //image:glm53_0906 --platforms={{platform}} --extra_execution_platforms={{platform}} --spawn_strategy=local --disk_cache=.bazel-cache --incompatible_strict_action_env "$@"

# Run the local Docker-backed image contract.
test *args:
    bazel --ignore_all_rc_files test //tests/image:glm53_0906_contract --platforms={{platform}} --extra_execution_platforms={{platform}} --spawn_strategy=local --disk_cache=.bazel-cache --incompatible_strict_action_env --test_output=errors "$@"

# Load the image into the local container daemon without starting it.
load *args:
    bazel --ignore_all_rc_files run //image:glm53_0906_load --platforms={{platform}} --extra_execution_platforms={{platform}} --spawn_strategy=local --disk_cache=.bazel-cache --incompatible_strict_action_env "$@"
