# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "vllmb12x_runtime_config", Path(__file__).with_name("vllmb12x-runtime-config.py")
)
INVENTORY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = INVENTORY
SPEC.loader.exec_module(INVENTORY)


class RuntimeConfigInventoryTest(unittest.TestCase):
    def test_scans_registered_and_direct_environment_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "vllm").mkdir()
            (root / "vllm/envs.py").write_text(
                'environment_variables = {\n'
                '    "VLLM_REGISTERED": lambda: os.getenv("VLLM_REGISTERED", "on"),\n'
                '}\n'
            )
            (root / "vllm/direct.py").write_text(
                'import os\nvalue = os.environ["VLLM_DIRECT"]\n'
                'additional_config = config.additional_config\n'
                'backend = additional_config.get("ple_table_memory", "device")\n'
            )

            scanned = INVENTORY.scan(root)
            controls = INVENTORY.distinct(scanned["environment"])

        self.assertEqual(controls["VLLM_REGISTERED"].default, "'on'")
        self.assertEqual(controls["VLLM_REGISTERED"].accepted, "environment string")
        self.assertEqual(controls["VLLM_DIRECT"].default, "unset")
        self.assertEqual(
            scanned["additional_config"][0].name,
            "ple_table_memory",
        )

    def test_reports_added_and_changed_defaults_only(self):
        stock = {
            "environment": [
                INVENTORY.Control("VLLM_STOCK", "vllm/envs.py", 1, "1", "environment string"),
                INVENTORY.Control("VLLM_CHANGED", "vllm/envs.py", 2, "1", "environment string"),
            ],
            "cli": [],
            "additional_config": [],
            "config": [],
        }
        integration = {
            "environment": [
                INVENTORY.Control("VLLM_STOCK", "vllm/envs.py", 1, "1", "environment string"),
                INVENTORY.Control("VLLM_CHANGED", "vllm/envs.py", 2, "2", "environment string"),
                INVENTORY.Control("VLLM_NEW", "vllm/envs.py", 3, "unset", "environment string"),
            ],
            "cli": [],
            "additional_config": [],
            "config": [],
        }

        controls = INVENTORY.vllm_controls(stock, integration)

        self.assertEqual([control.name for _, control in controls], ["VLLM_CHANGED", "VLLM_NEW"])

    def test_extracts_source_descriptions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "vllm").mkdir()
            (root / "vllm/args.py").write_text(
                "parser.add_argument('--backend', help='Select the execution backend.')\n"
            )
            (root / "vllm/config").mkdir()
            (root / "vllm/config/fields.py").write_text(
                "class Config:\n"
                "    backend: str = 'auto'\n"
                "    '''Select the execution backend for requests.'''\n"
            )
            (root / "vllm/envs.py").write_text(
                "# Keep the worker-side transport aligned with the serving group.\n"
                "environment_variables = {'VLLM_TRANSPORT': lambda: os.getenv('VLLM_TRANSPORT', 'nccl')}\n"
            )

            scanned = INVENTORY.scan(root)

        self.assertEqual(scanned["cli"][0].description, "Select the execution backend.")
        self.assertEqual(scanned["config"][0].description, "Select the execution backend for requests.")
        self.assertEqual(
            INVENTORY.distinct(scanned["environment"])["VLLM_TRANSPORT"].description,
            "Keep the worker-side transport aligned with the serving group.",
        )

    def test_reads_exact_and_prefix_description_mixin_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixin.yaml"
            path.write_text(
                "descriptions:\n"
                "  VLLM_EXACT: \"Exact source override.\"\n"
                "  B12X_*: \"B12X source override.\"\n"
            )

            descriptions = INVENTORY.description_mixin(path)

        self.assertEqual(
            INVENTORY.mixin_description("VLLM_EXACT", descriptions),
            "Exact source override.",
        )
        self.assertEqual(
            INVENTORY.mixin_description("B12X_SETTING", descriptions),
            "B12X source override.",
        )

    def test_runtime_mixin_expands_tuning_controls(self):
        descriptions = INVENTORY.description_mixin(
            Path(__file__).with_name("vllmb12x-runtime-config.yaml")
        )

        self.assertEqual(
            INVENTORY.mixin_description("B12X_MHC_PREFILL_TMA_TILE_M", descriptions),
            "Set the M tile dimension for the MHC prefill TMA kernel.",
        )
        self.assertEqual(
            INVENTORY.mixin_description("B12X_PCIE_ONESHOT_PUSH", descriptions),
            "Enable the one-shot PCIe push transport, which writes each rank's input into every peer's eager slot.",
        )

    def test_renders_compact_source_reference(self):
        control = INVENTORY.Control(
            "--backend",
            "vllm/args.py",
            1,
            "'auto'",
            "['auto', 'b12x']",
            "Select the execution backend.",
        )
        empty = {surface: [] for surface in ("environment", "cli", "additional_config", "config")}
        rendered = INVENTORY.render("stock", "vllm", "b12x", [("cli", control)], empty)

        self.assertIn('<table class="runtime-config-table">', rendered)
        self.assertIn("<th scope=\"col\">Setting</th>", rendered)
        self.assertIn("<small class=\"runtime-config-meta\">cli<br>default: <code>&#x27;auto&#x27;</code></small>", rendered)
        self.assertIn("<th scope=\"col\">Accepted / source</th>", rendered)
        self.assertIn("Select the execution backend.", rendered)
        self.assertIn("https://docs.vllm.ai/en/latest/configuration/", rendered)
        self.assertNotIn("Classification", rendered)
        self.assertNotIn("Provenance", rendered)
        self.assertNotIn("Qwen recipe", rendered)

    def test_source_description_precedes_mixin_fallback(self):
        control = INVENTORY.Control(
            "VLLM_EXACT",
            "vllm/envs.py",
            1,
            "'source'",
            "environment string",
            "Source description.",
        )

        self.assertEqual(
            INVENTORY.control_description(
                "environment",
                control,
                [control],
                {"VLLM_EXACT": "Mixin description."},
            ),
            "Source description.",
        )

    def test_reports_changed_accepted_values(self):
        stock = {
            "environment": [],
            "cli": [INVENTORY.Control("--kernel", "vllm/args.py", 1, "auto", "['auto']")],
            "additional_config": [],
            "config": [],
        }
        integration = {
            "environment": [],
            "cli": [INVENTORY.Control("--kernel", "vllm/args.py", 1, "auto", "['auto', 'b12x']")],
            "additional_config": [],
            "config": [],
        }

        controls = INVENTORY.vllm_controls(stock, integration)

        self.assertEqual([control.name for _, control in controls], ["--kernel"])

    def test_collects_indirect_environment_control_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b12x").mkdir()
            (root / "b12x/controls.py").write_text(
                'NAMES = ("B12X_INDIRECT",)\n'
                'for name in NAMES:\n'
                '    value = os.getenv(name)\n'
            )

            controls = INVENTORY.distinct(INVENTORY.scan(root)["environment"])

        self.assertIn("B12X_INDIRECT", controls)

if __name__ == "__main__":
    unittest.main()
