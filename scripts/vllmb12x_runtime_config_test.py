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

    def test_classifies_b12x_vllm_named_controls_as_b12x(self):
        self.assertEqual(
            INVENTORY.provenance("VLLM_B12X_TIMING", "b12x/integration/vllm/plugin.py"),
            "B12X integration",
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
