from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PublicPackageTest(unittest.TestCase):
    def test_expected_public_files_exist(self) -> None:
        expected = [
            "README.md",
            ".gitignore",
            "requirements.txt",
            "config/nox_config.example.json",
            "docs/workflow_overview.md",
            "docs/github_publish_guide.md",
            "docs/data_contract.md",
            "examples/schema_sample.csv",
            "scripts/stage00p_raw_intake.py",
            "scripts/stage03_feature_builder.py",
            "scripts/stage04_temporal_split.py",
            "scripts/stage05_to_07_baseline.py",
            "scripts/run_share_pipeline.ps1",
        ]
        for relative in expected:
            with self.subTest(relative=relative):
                self.assertTrue((ROOT / relative).exists())

    def test_public_text_files_do_not_contain_private_markers(self) -> None:
        banned = [
            "C:" + "\\Users",
            "Served" + " Data",
            "2025" + "0811",
            "2025" + "0825",
            "References" + "/01",
            "." + "venv-cantera",
        ]
        checked_suffixes = {".md", ".py", ".ps1", ".json", ".txt", ".csv"}
        for path in ROOT.rglob("*"):
            if ".git" in path.parts or path.suffix.lower() not in checked_suffixes:
                continue
            text = path.read_text(encoding="utf-8")
            for marker in banned:
                with self.subTest(path=path.relative_to(ROOT), marker=marker):
                    self.assertNotIn(marker, text)


if __name__ == "__main__":
    unittest.main()
