#!/usr/bin/env python3
"""Generate the VLLMB12X runtime-control delta from pinned source trees."""

from __future__ import annotations

import argparse
import ast
import hashlib
import re
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


VLLM_PREFIXES = ("VLLM_", "B12X_")
ENVIRONMENT_NAME = re.compile(r"(?:VLLM|B12X)_[A-Z0-9_]+$")
SOURCE_SUFFIXES = {".py", ".pyi"}
SKIP_DIRECTORIES = {
    ".buildkite",
    ".git",
    ".venv",
    "benchmarks",
    "build",
    "docs",
    "scripts",
    "tests",
    "tools",
}


@dataclass(frozen=True, order=True)
class Control:
    name: str
    source: str
    line: int
    default: str
    accepted: str


def source_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.suffix not in SOURCE_SUFFIXES or any(part in SKIP_DIRECTORIES for part in path.parts):
            continue
        yield path


def string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def environment_name(node: ast.AST | None) -> str | None:
    value = string(node)
    return value if value and ENVIRONMENT_NAME.fullmatch(value) else None


def expression(node: ast.AST | None) -> str:
    if node is None:
        return "unset"
    try:
        return ast.unparse(node)
    except Exception:
        return "source expression"


def is_environment_read(node: ast.AST) -> bool:
    if not isinstance(node, ast.Attribute):
        return False
    if node.attr == "getenv" and isinstance(node.value, ast.Name) and node.value.id == "os":
        return True
    if node.attr == "get" and isinstance(node.value, ast.Attribute) and node.value.attr == "environ":
        return True
    if node.attr == "get" and isinstance(node.value, ast.Name) and node.value.id == "environ":
        return True
    return False


class Scanner(ast.NodeVisitor):
    def __init__(self, root: Path, path: Path):
        self.root = root
        self.path = path
        self.environment: list[Control] = []
        self.cli: list[Control] = []
        self.additional_config: list[Control] = []
        self.additional_config_aliases: set[str] = set()
        self.environment_name_groups: dict[str, tuple[str, ...]] = {}

    @property
    def source(self) -> str:
        return self.path.relative_to(self.root).as_posix()

    def visit_Call(self, node: ast.Call) -> None:
        if is_environment_read(node.func) and node.args:
            name = environment_name(node.args[0])
            if name:
                self.environment.append(Control(name, self.source, node.lineno, expression(node.args[1] if len(node.args) > 1 else None), "environment string"))
            elif isinstance(node.args[0], ast.Name):
                for indirect_name in self.environment_name_groups.get(node.args[0].id, ()):
                    self.environment.append(Control(indirect_name, self.source, node.lineno, "source expression", "environment string"))
        if isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument" and node.args:
            name = string(node.args[0])
            if name and name.startswith("--"):
                default = "unset"
                for keyword in node.keywords:
                    if keyword.arg == "default":
                        default = expression(keyword.value)
                choices = next((expression(keyword.value) for keyword in node.keywords if keyword.arg == "choices"), None)
                arg_type = next((expression(keyword.value) for keyword in node.keywords if keyword.arg == "type"), None)
                self.cli.append(Control(name, self.source, node.lineno, default, choices or arg_type or "string"))
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"get", "pop", "__getitem__"} and node.args:
            name = string(node.args[0])
            value = node.func.value
            if name and (
                (isinstance(value, ast.Attribute) and value.attr == "additional_config")
                or (isinstance(value, ast.Name) and value.id in self.additional_config_aliases)
            ):
                self.additional_config.append(Control(name, self.source, node.lineno, expression(node.args[1] if len(node.args) > 1 else None), "JSON value"))
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if isinstance(node.value, ast.Attribute) and node.value.attr == "additional_config":
            self.additional_config_aliases.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
        if isinstance(node.value, (ast.Tuple, ast.List, ast.Set)):
            names = tuple(
                value.value
                for value in node.value.elts
                if isinstance(value, ast.Constant)
                and isinstance(value.value, str)
                and ENVIRONMENT_NAME.fullmatch(value.value)
            )
            if names:
                self.environment_name_groups.update(
                    {target.id: names for target in node.targets if isinstance(target, ast.Name)}
                )
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        if isinstance(node.target, ast.Name) and isinstance(node.iter, ast.Name):
            names = self.environment_name_groups.get(node.iter.id)
            if names:
                self.environment_name_groups[node.target.id] = names
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        value = node.value
        is_environ = (
            isinstance(value, ast.Attribute)
            and value.attr == "environ"
            and isinstance(value.value, ast.Name)
            and value.value.id == "os"
        ) or (isinstance(value, ast.Name) and value.id == "environ")
        name = environment_name(node.slice)
        if is_environ and name:
            self.environment.append(Control(name, self.source, node.lineno, "unset", "environment string"))
        if isinstance(value, ast.Attribute) and value.attr == "additional_config" and (name := string(node.slice)):
            self.additional_config.append(Control(name, self.source, node.lineno, "unset", "JSON value"))
        self.generic_visit(node)


def registered_environment(root: Path) -> list[Control]:
    path = root / "vllm/envs.py"
    if not path.is_file():
        return []
    lines = path.read_text().splitlines()
    controls: list[Control] = []
    for index, line in enumerate(lines, start=1):
        match = re.match(r'\s*"((?:VLLM|B12X)_[A-Z0-9_]+)"\s*:\s*(.*)', line)
        if not match:
            continue
        name, value = match.groups()
        block = " ".join(lines[index - 1:min(len(lines), index + 8)])
        default_match = re.search(r'os\.(?:getenv|environ\.get)\([^,]+,\s*([^\)]+)\)', block)
        default = default_match.group(1).strip() if default_match else "source expression"
        controls.append(Control(name, "vllm/envs.py", index, default, "environment string"))
    return controls


def scan(root: Path) -> dict[str, list[Control]]:
    result: dict[str, list[Control]] = defaultdict(list)
    for path in source_files(root):
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        scanner = Scanner(root, path)
        scanner.visit(tree)
        result["environment"].extend(scanner.environment)
        result["cli"].extend(scanner.cli)
        result["additional_config"].extend(scanner.additional_config)
    result["environment"].extend(registered_environment(root))
    result["config"].extend(config_fields(root))
    return result


class ConfigFieldScanner(ast.NodeVisitor):
    def __init__(self, root: Path, path: Path):
        self.root = root
        self.path = path
        self.classes: list[str] = []
        self.controls: list[Control] = []

    @property
    def source(self) -> str:
        return self.path.relative_to(self.root).as_posix()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.classes.append(node.name)
        self.generic_visit(node)
        self.classes.pop()

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if self.classes and isinstance(node.target, ast.Name):
            name = ".".join((*self.classes, node.target.id))
            self.controls.append(Control(
                name,
                self.source,
                node.lineno,
                expression(node.value),
                expression(node.annotation),
            ))
        self.generic_visit(node)


def config_fields(root: Path) -> list[Control]:
    directory = root / "vllm/config"
    if not directory.is_dir():
        return []
    controls: list[Control] = []
    for path in source_files(directory):
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        scanner = ConfigFieldScanner(root, path)
        scanner.visit(tree)
        controls.extend(scanner.controls)
    return controls


def distinct(controls: Iterable[Control]) -> dict[str, Control]:
    result: dict[str, Control] = {}
    def certainty(control: Control) -> int:
        if control.default == "source expression":
            return 2
        if control.default == "unset":
            return 1
        return 0

    for control in sorted(controls, key=lambda value: (value.name, certainty(value), value.source, value.line)):
        result.setdefault(control.name, control)
    return result


def vllm_controls(stock: dict[str, list[Control]], integration: dict[str, list[Control]]) -> list[tuple[str, Control]]:
    rows: list[tuple[str, Control]] = []
    for surface in ("environment", "cli", "additional_config", "config"):
        baseline = distinct(stock[surface])
        for name, control in distinct(integration[surface]).items():
            if (
                name not in baseline
                or baseline[name].default != control.default
                or baseline[name].accepted != control.accepted
            ):
                rows.append((surface, control))
    return rows


def classification(name: str) -> str:
    lower = name.lower()
    if any(token in lower for token in ("debug", "dump", "trace", "log", "timing", "probe", "validate", "test")):
        return "diagnostic"
    if any(token in lower for token in ("autotune", "tune", "tile", "split", "threads", "ctas", "warps", "stages", "smem", "turbo", "prefetch", "grace", "spin", "park")):
        return "experimental tuning"
    if any(token in lower for token in ("cache", "compile", "disk", "backend", "roce", "pcie", "ple", "indexer")):
        return "serving control"
    return "runtime control"


def provenance(name: str, source: str) -> str:
    if source.startswith("b12x/") or name.startswith("B12X_"):
        return "B12X integration"
    if name.startswith("VLLM_SHM_BROADCAST_ADAPTIVE") or name == "VLLM_SHM_BROADCAST_WRITE_PARK_MAX_MS":
        return "vLLM #52917"
    if "qwen3_8" in name.lower() or "hc_prefill" in source:
        return "local-inference-lab/vLLM #779"
    return "local-inference-lab/vLLM"


def recipe_values(recipe: Path) -> dict[str, str]:
    if not recipe.is_file():
        return {}
    values: dict[str, str] = {}
    for line in recipe.read_text().splitlines():
        match = re.match(r"\s{4}([A-Z][A-Z0-9_]+):\s*(.*)", line)
        if match:
            values[match.group(1)] = match.group(2).strip() or "set"
    values["--additional-config"] = '{"ple_table_memory":"disk"}'
    values["--hf-overrides"] = "YaRN 4x M-RoPE override"
    values["--gdn-decode-kernel"] = "b12x"
    values["--linear-backend"] = "b12x"
    values["--moe-backend"] = "b12x"
    return values


def recipe_value(control: Control, values: dict[str, str]) -> str:
    if control.name in values:
        return values[control.name]
    if control.name.endswith(".moe_backend"):
        return values.get("--moe-backend", "not set by Qwen recipe")
    if control.name.endswith(".linear_backend"):
        return values.get("--linear-backend", "not set by Qwen recipe")
    if control.name.endswith(".gdn_decode_kernel"):
        return values.get("--gdn-decode-kernel", "not set by Qwen recipe")
    return "not set by Qwen recipe"


def effect(surface: str, control: Control) -> str:
    name = control.name
    if name.startswith("B12X_COMPILE_"):
        return "Controls B12X generated-kernel compilation cache behavior."
    if name.startswith("B12X_ROCE_"):
        return "Controls B12X one-shot RoCE collective setup or wait behavior."
    if name.startswith("B12X_DENSE_"):
        return "Controls B12X dense GEMM specialization behavior."
    if name.startswith("B12X_MOE_"):
        return "Controls B12X MoE kernel selection or tuning."
    if name.startswith("B12X_PLE_") or name.startswith("VLLM_PLE_"):
        return "Controls the PLE table storage or execution path."
    if name.startswith("VLLM_QWEN3_8_"):
        return "Controls Qwen3.8 Flash Next execution behavior."
    if name.startswith("VLLM_SHM_BROADCAST_"):
        return "Controls adaptive shared-memory broadcast waiting."
    if surface == "environment":
        return "Read at runtime by the cited source."
    if surface == "cli":
        return "vLLM command-line option declared by the cited source."
    if surface == "config":
        return "vLLM configuration field consumed by the cited source."
    return "B12X/vLLM model additional_config key read by the cited source."


def row(surface: str, control: Control, values: dict[str, str]) -> str:
    recipe = recipe_value(control, values)
    return "| `{}` | {} | `{}` | {} | {} | {} | {}:{} | {} | {} |".format(
        control.name,
        surface,
        control.default.replace("|", "\\|"),
        control.accepted.replace("|", "\\|"),
        classification(control.name),
        provenance(control.name, control.source),
        control.source,
        control.line,
        effect(surface, control),
        recipe.replace("|", "\\|"),
    )


def render(stock_commit: str, vllm_commit: str, b12x_commit: str, controls: list[tuple[str, Control]], b12x: dict[str, list[Control]], values: dict[str, str]) -> str:
    rows_by_name: dict[tuple[str, str], tuple[str, Control]] = {}
    for surface, control in controls:
        rows_by_name.setdefault((surface, control.name), (surface, control))
    # B12X owns the implementation and documentation for controls it also
    # exposes through vLLM's integration plugin.
    for control in distinct(b12x["environment"]).values():
        rows_by_name[("environment", control.name)] = ("environment", control)
    rows = sorted(rows_by_name.values(), key=lambda item: (item[0], item[1].name))
    digest = hashlib.sha256("\n".join(f"{surface}:{control}" for surface, control in rows).encode()).hexdigest()
    body = [
        "# VLLMB12X Runtime Configuration",
        "",
        "Generated by `scripts/vllmb12x-runtime-config.py`. Do not edit by hand.",
        "",
        "## Compared Sources",
        "",
        f"- Stock vLLM merge-base: `{stock_commit}`.",
        f"- Locked randomvariable/vLLM integration before profile patches: `{vllm_commit}`.",
        f"- Locked B12X integration: `{b12x_commit}`.",
        "- The build applies `third_party/vllm_shm_broadcast_spin_grace.patch` to the locked vLLM tree. Its controls are listed as vLLM #52917.",
        "",
        "This inventory contains runtime controls added or whose default changed relative to the stock vLLM merge-base, plus every B12X environment control consumed by the locked B12X source. Static inspection records each environment value as a string because the source performs its own parsing. `unset` means the cited read has no source default. `source expression` identifies an indirect control name or nonliteral source default. Recipe values describe the Qwen recipe only and are not recommendations for controls it leaves unset.",
        "",
        "## Controls",
        "",
        "| Name | Surface | Default | Accepted values or type | Classification | Provenance | Source | Effect | Qwen recipe |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    body.extend(row(surface, control, values) for surface, control in rows)
    body.extend([
        "",
        "## Semantic Extensions",
        "",
        "- `--hf-overrides` retains dictionary YaRN and rope values when vLLM builds an in-model MTP draft `ModelConfig` (local-inference-lab/vLLM #777). The Qwen recipe sets a 4x YaRN M-RoPE dictionary so the draft and target share the 1,048,576-token geometry.",
        "- `VLLM_QWEN3_8_PREFILL_COALESCE` requires B12X #386's prepared PLE checkpoint export. The Qwen recipe does not set this experimental control.",
        "- The adaptive shared-memory controls alter wait behavior only and are excluded from vLLM compile factors. They do not select kernels or invalidate compiled graphs.",
        "",
        "## Generator Identity",
        "",
        f"`{digest}`",
        "",
    ])
    return "\n".join(body)


def apply_patch(root: Path, patch: Path) -> Path:
    destination = Path(tempfile.mkdtemp(prefix="vllmb12x-runtime-config-")) / "vllm"
    subprocess.run(["cp", "-a", root, destination], check=True)
    subprocess.run(
        ["patch", "-d", destination, "-p1", "-i", patch.resolve()],
        check=True,
        capture_output=True,
        text=True,
    )
    return destination


def git_revision(root: Path) -> str:
    return subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock-root", type=Path, required=True)
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--b12x-root", type=Path, required=True)
    parser.add_argument("--vllm-patch", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    integration = apply_patch(args.vllm_root, args.vllm_patch)
    content = render(
        git_revision(args.stock_root),
        git_revision(args.vllm_root),
        git_revision(args.b12x_root),
        vllm_controls(scan(args.stock_root), scan(integration)),
        scan(args.b12x_root),
        recipe_values(args.recipe),
    )
    if args.check:
        if not args.output.is_file() or args.output.read_text() != content:
            raise SystemExit(f"{args.output} is stale; regenerate it")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content)


if __name__ == "__main__":
    main()
