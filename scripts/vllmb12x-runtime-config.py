#!/usr/bin/env python3
"""Generate the VLLMB12X runtime-control delta from pinned source trees."""

from __future__ import annotations

import argparse
import ast
import io
import hashlib
import html
import json
import os
import re
import subprocess
import tarfile
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ENVIRONMENT_NAME = re.compile(r"(?:VLLM|B12X)_[A-Z0-9_]+$")
SOURCE_SUFFIXES = {".py", ".pyi"}
SKIP_DIRECTORIES = {
    ".agents",
    ".buildkite",
    ".git",
    ".omp",
    ".venv",
    ".omp-test-venv",
    "assets",
    "benchmarks",
    "build",
    "coverage",
    "docs",
    "examples",
    "external",
    "node_modules",
    "output",
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
    description: str = ""


def normalized_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def comment_description(lines: list[str], line: int) -> str:
    comments: list[str] = []
    index = line - 2
    if index < 0:
        return ""
    index = min(index, len(lines) - 1)
    while index >= 0:
        stripped = lines[index].strip()
        if not stripped.startswith("#"):
            break
        comments.append(stripped[1:].strip())
        index -= 1
    return normalized_text(" ".join(reversed(comments)))


def string_docstring(node: ast.AST | None) -> str:
    if isinstance(node, ast.Expr):
        return normalized_text(string(node.value))
    return ""


def yaml_scalar(value: str) -> str:
    value = value.strip()
    if value.startswith('"'):
        parsed = json.loads(value)
    elif value.startswith("'"):
        parsed = ast.literal_eval(value)
    else:
        parsed = value
    if not isinstance(parsed, str):
        raise ValueError(f"description mixin values must be strings: {value}")
    return normalized_text(parsed)


def description_mixin(path: Path) -> dict[str, str]:
    """Read the flat YAML description overlay without adding a Python dependency."""
    if not path.is_file():
        return {}
    descriptions: dict[str, str] = {}
    in_descriptions = False
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not line.startswith("  "):
            if stripped != "descriptions:":
                raise ValueError(f"{path}:{line_number}: expected descriptions:")
            in_descriptions = True
            continue
        if not in_descriptions or ":" not in stripped:
            raise ValueError(f"{path}:{line_number}: expected a description mapping")
        name, value = stripped.split(":", 1)
        name = yaml_scalar(name)
        if name in descriptions:
            raise ValueError(f"{path}:{line_number}: duplicate description for {name}")
        descriptions[name] = yaml_scalar(value)
    return descriptions


def source_files(root: Path) -> Iterable[Path]:
    for directory, directories, files in os.walk(root):
        directories[:] = [
            child
            for child in directories
            if child not in SKIP_DIRECTORIES and not child.startswith(".venv")
        ]
        for name in files:
            path = Path(directory, name)
            if path.suffix in SOURCE_SUFFIXES:
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
        self.lines = path.read_text().splitlines()

    @property
    def source(self) -> str:
        return self.path.relative_to(self.root).as_posix()

    def visit_Call(self, node: ast.Call) -> None:
        if is_environment_read(node.func) and node.args:
            name = environment_name(node.args[0])
            if name:
                self.environment.append(Control(
                    name,
                    self.source,
                    node.lineno,
                    expression(node.args[1] if len(node.args) > 1 else None),
                    "environment string",
                    comment_description(self.lines, node.lineno),
                ))
            elif isinstance(node.args[0], ast.Name):
                for indirect_name in self.environment_name_groups.get(node.args[0].id, ()):
                    self.environment.append(Control(
                        indirect_name,
                        self.source,
                        node.lineno,
                        "source expression",
                        "environment string",
                        comment_description(self.lines, node.lineno),
                    ))
        if isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument" and node.args:
            name = string(node.args[0])
            if name and name.startswith("--"):
                default = "unset"
                for keyword in node.keywords:
                    if keyword.arg == "default":
                        default = expression(keyword.value)
                choices = next((expression(keyword.value) for keyword in node.keywords if keyword.arg == "choices"), None)
                arg_type = next((expression(keyword.value) for keyword in node.keywords if keyword.arg == "type"), None)
                help_text = next((string(keyword.value) for keyword in node.keywords if keyword.arg == "help"), None)
                self.cli.append(Control(
                    name,
                    self.source,
                    node.lineno,
                    default,
                    choices or arg_type or "string",
                    normalized_text(help_text),
                ))
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"get", "pop", "__getitem__"} and node.args:
            name = string(node.args[0])
            value = node.func.value
            if name and (
                (isinstance(value, ast.Attribute) and value.attr == "additional_config")
                or (isinstance(value, ast.Name) and value.id in self.additional_config_aliases)
            ):
                self.additional_config.append(Control(
                    name,
                    self.source,
                    node.lineno,
                    expression(node.args[1] if len(node.args) > 1 else None),
                    "JSON value",
                    comment_description(self.lines, node.lineno),
                ))
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
            self.environment.append(Control(
                name,
                self.source,
                node.lineno,
                "unset",
                "environment string",
                comment_description(self.lines, node.lineno),
            ))
        if isinstance(value, ast.Attribute) and value.attr == "additional_config" and (name := string(node.slice)):
            self.additional_config.append(Control(
                name,
                self.source,
                node.lineno,
                "unset",
                "JSON value",
                comment_description(self.lines, node.lineno),
            ))
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
        controls.append(Control(
            name,
            "vllm/envs.py",
            index,
            default,
            "environment string",
            comment_description(lines, index),
        ))
    return controls


def scan(root: Path) -> dict[str, list[Control]]:
    result: dict[str, list[Control]] = defaultdict(list)
    package_root = root / "b12x"
    scan_root = package_root if package_root.is_dir() else root
    for path in source_files(scan_root):
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except (RecursionError, SyntaxError, UnicodeDecodeError):
            continue
        scanner = Scanner(root, path)
        try:
            scanner.visit(tree)
        except RecursionError:
            continue
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
        for index, child in enumerate(node.body):
            if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                docstring = string_docstring(node.body[index + 1]) if index + 1 < len(node.body) else ""
                self.controls.append(Control(
                    ".".join((*self.classes, child.target.id)),
                    self.source,
                    child.lineno,
                    expression(child.value),
                    expression(child.annotation),
                    docstring,
                ))
            elif isinstance(child, ast.ClassDef):
                self.visit(child)
        self.classes.pop()


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
        if not control.description:
            return 3
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


def mixin_description(name: str, descriptions: dict[str, str]) -> str:
    if name in descriptions:
        return descriptions[name]
    for pattern, description in sorted(descriptions.items(), key=lambda item: -len(item[0])):
        if pattern.endswith("*") and name.startswith(pattern[:-1]):
            return description
    return ""


def control_description(
    surface: str,
    control: Control,
    controls: Iterable[Control],
    descriptions: dict[str, str],
) -> str:
    if control.description:
        return control.description
    if surface == "cli":
        field_name = control.name.removeprefix("--").replace("-", "_")
        for candidate in controls:
            if candidate.name.rsplit(".", 1)[-1] == field_name and candidate.description:
                return candidate.description
    exact = descriptions.get(control.name)
    if exact:
        return exact
    pattern = mixin_description(control.name, descriptions)
    if pattern:
        return pattern
    if surface == "environment":
        return "Environment variable read by the cited source."
    if surface == "cli":
        return "Command-line option declared by the cited source."
    if surface == "config":
        return "Configuration field consumed by the cited source."
    return "Additional configuration key read by the cited source."


def html_text(value: str) -> str:
    escaped = html.escape(normalized_text(value), quote=False)
    return re.sub(r"`{1,2}([^`]+)`{1,2}", r"<code>\1</code>", escaped)


def row(
    surface: str,
    control: Control,
    controls: Iterable[Control],
    descriptions: dict[str, str],
) -> str:
    description = control_description(surface, control, controls, descriptions)
    name = html.escape(control.name, quote=True)
    surface_text = html.escape(surface, quote=False)
    default = html.escape(normalized_text(control.default), quote=True)
    accepted = html.escape(normalized_text(control.accepted), quote=True)
    source = html.escape(f"{control.source}:{control.line}", quote=True)
    return (
        "<tr>"
        f"<td><code>{name}</code><small class=\"runtime-config-meta\">"
        f"{surface_text}<br>default: <code>{default}</code></small></td>"
        f"<td>{html_text(description)}</td>"
        f"<td><code>{accepted}</code><small class=\"runtime-config-meta\">"
        f"<code>{source}</code></small></td>"
        "</tr>"
    )


def render(
    stock_commit: str,
    vllm_commit: str,
    b12x_commit: str,
    controls: list[tuple[str, Control]],
    b12x: dict[str, list[Control]],
    applied_patches: Iterable[Path] = (),
    descriptions: dict[str, str] | None = None,
) -> str:
    descriptions = descriptions or {}
    rows_by_name: dict[tuple[str, str], tuple[str, Control]] = {}
    for surface, control in controls:
        rows_by_name.setdefault((surface, control.name), (surface, control))
    # B12X owns the implementation and documentation for controls it also
    # exposes through vLLM's integration plugin.
    for control in distinct(b12x["environment"]).values():
        rows_by_name[("environment", control.name)] = ("environment", control)
    rows = sorted(rows_by_name.values(), key=lambda item: (item[0], item[1].name))
    all_controls = [control for _, control in rows] + list(distinct(b12x["config"]).values())
    digest = hashlib.sha256(
        "\n".join(
            f"{surface}:{control.name}:{control_description(surface, control, all_controls, descriptions)}"
            for surface, control in rows
        ).encode()
    ).hexdigest()
    patch_lines = [
        f"- The build applies `{patch.as_posix()}` to the locked vLLM tree."
        for patch in applied_patches
    ]
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
        *patch_lines,
        "",
        "This inventory contains runtime controls added or whose default or accepted values changed relative to the stock vLLM merge-base, plus every B12X environment control consumed by the locked B12X source. Descriptions come from source help text, field docstrings, source comments, or the checked-in description mixin when the source has no prose. Static inspection records each environment value as a string because the source performs its own parsing. `unset` means the cited read has no source default. `source expression` identifies an indirect control name or nonliteral source default.",
        "For regular vLLM configuration, see the [vLLM configuration reference](https://docs.vllm.ai/en/latest/configuration/).",
        "",
        "## Controls",
        "",
        '<table class="runtime-config-table">',
        "<thead>",
        "<tr><th scope=\"col\">Setting</th><th scope=\"col\">Description</th><th scope=\"col\">Accepted / source</th></tr>",
        "</thead>",
        "<tbody>",
    ]
    body.extend(row(surface, control, all_controls, descriptions) for surface, control in rows)
    body.extend([
        "</tbody>",
        "</table>",
        "",
        "## Semantic Extensions",
        "",
        "- `--hf-overrides` retains dictionary YaRN and rope values when vLLM builds an in-model MTP draft `ModelConfig` (local-inference-lab/vLLM #777).",
        "- `VLLM_QWEN3_8_PREFILL_COALESCE` requires B12X #386's prepared PLE checkpoint export and vLLM 5dd5bd5dde76 or later, which wires the prefill checkpoint blocks into the NVIDIA GDN decoder. On earlier cuts that carry the coalesce feature from 61f93c53ee, setting it aborts boot with \"all mamba groups must share cache scheduling parameters\".",
        "- Never set `CUDA_LAUNCH_BLOCKING`. b12x does not function with synchronous kernel launches, so the variable is not a usable debug lever on this stack.",
        "- The adaptive shared-memory controls alter wait behavior only and are excluded from vLLM compile factors. They do not select kernels or invalidate compiled graphs.",
        "",
        "## Generator Identity",
        "",
        f"`{digest}`",
        "",
    ])
    return "\n".join(body)


def apply_patches(root: Path, patches: Iterable[Path]) -> Path:
    destination = Path(tempfile.mkdtemp(prefix="vllmb12x-runtime-config-")) / "vllm"
    archive = subprocess.run(
        ["git", "-C", root, "archive", "--format=tar", "HEAD"],
        check=True,
        capture_output=True,
    )
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        tar.extractall(destination, filter="tar")
    for patch in patches:
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
    parser.add_argument("--vllm-patch", type=Path, action="append", default=[])
    parser.add_argument(
        "--description-mixin",
        type=Path,
        default=Path(__file__).with_name("vllmb12x-runtime-config.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    integration = apply_patches(args.vllm_root, args.vllm_patch)
    content = render(
        git_revision(args.stock_root),
        git_revision(args.vllm_root),
        git_revision(args.b12x_root),
        vllm_controls(scan(args.stock_root), scan(integration)),
        scan(args.b12x_root),
        args.vllm_patch,
        description_mixin(args.description_mixin),
    )
    if args.check:
        if not args.output.is_file() or args.output.read_text() != content:
            raise SystemExit(f"{args.output} is stale; regenerate it")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content)


if __name__ == "__main__":
    main()
