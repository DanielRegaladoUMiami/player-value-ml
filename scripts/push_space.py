"""Push the Gradio Space app to Hugging Face Spaces.

Validates BEFORE push:
  - app.py is valid Python
  - requirements.txt lists gradio + lightgbm + pandas
  - README.md has HF Spaces YAML frontmatter
  - The Space needs the LightGBM model + the test parquet — we copy them
    into the staging dir so the Space is self-contained at boot.

Usage:
    python -m scripts.push_space --repo DanielRegaladoCardoso/player-value-ml
"""
from __future__ import annotations

import argparse
import ast
import logging
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
SPACE = ROOT / "space"
LGBM_DIR = ROOT / "results" / "models" / "lgbm"
TEST_PARQUET = ROOT / "data" / "features" / "test.parquet"


def validate() -> None:
    for p in (SPACE / "app.py", SPACE / "requirements.txt",
              SPACE / "README.md", LGBM_DIR / "model.lgb", TEST_PARQUET):
        assert p.exists(), f"FAIL: {p} missing"
    ast.parse((SPACE / "app.py").read_text())
    logger.info("app.py: valid Python syntax")

    req = (SPACE / "requirements.txt").read_text()
    for pkg in ("gradio", "lightgbm", "pandas"):
        assert pkg in req, f"FAIL: requirements.txt missing {pkg}"
    logger.info("requirements.txt: gradio + lightgbm + pandas ok")

    readme = (SPACE / "README.md").read_text()
    assert readme.startswith("---") and "sdk: gradio" in readme, \
        "FAIL: README.md missing HF Spaces frontmatter"
    logger.info("README.md: HF Spaces frontmatter present")


def assemble(staging: Path) -> None:
    """Bundle app.py, requirements, README, model + test parquet."""
    shutil.copy2(SPACE / "app.py", staging / "app.py")
    shutil.copy2(SPACE / "requirements.txt", staging / "requirements.txt")
    shutil.copy2(SPACE / "README.md", staging / "README.md")
    # The app expects the model file relative to its own dir.
    (staging / "results" / "models" / "lgbm").mkdir(parents=True, exist_ok=True)
    shutil.copy2(LGBM_DIR / "model.lgb",
                 staging / "results" / "models" / "lgbm" / "model.lgb")
    shutil.copy2(LGBM_DIR / "feature_importance.csv",
                 staging / "results" / "models" / "lgbm" / "feature_importance.csv")
    (staging / "data" / "features").mkdir(parents=True, exist_ok=True)
    shutil.copy2(TEST_PARQUET, staging / "data" / "features" / "test.parquet")
    files = sorted(p.relative_to(staging) for p in staging.rglob("*") if p.is_file())
    logger.info("Staged %d files: %s", len(files), [str(f) for f in files])


def push(staging: Path, repo: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    try:
        whoami = api.whoami()
    except Exception as e:
        raise SystemExit(f"FAIL: HF auth not set. Run `hf auth login`. {e}")
    logger.info("HF auth OK as %s", whoami.get("name"))

    api.create_repo(repo_id=repo, repo_type="space", space_sdk="gradio",
                    exist_ok=True)
    api.upload_folder(folder_path=str(staging), repo_id=repo, repo_type="space")
    logger.info("Pushed to https://huggingface.co/spaces/%s", repo)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="DanielRegaladoCardoso/player-value-ml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    validate()
    with tempfile.TemporaryDirectory() as td:
        staging = Path(td)
        assemble(staging)
        if args.dry_run:
            logger.info("DRY RUN: validations passed, staged at %s. Skipping push.", staging)
            return
        push(staging, args.repo)


if __name__ == "__main__":
    main()
