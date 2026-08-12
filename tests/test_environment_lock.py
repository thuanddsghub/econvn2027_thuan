"""Ensure the frozen environment lock covers every declared dependency."""

import re
import tomllib
from pathlib import Path


def requirement_name(requirement: str) -> str:
    """Return the normalized distribution name from a PEP 508 requirement."""
    name = re.split(r"[<>=!~;\[]", requirement, maxsplit=1)[0].strip()
    return name.lower().replace("_", "-")


def test_environment_lock_matches_declared_dependencies():
    config = tomllib.loads(Path("pyproject.toml").read_text())
    declared = {
        requirement_name(requirement)
        for requirement in (
            config["build-system"]["requires"]
            + config["project"]["dependencies"]
            + config["project"]["optional-dependencies"]["dev"]
        )
    }

    lock_lines = [
        line.strip()
        for line in Path("requirements-lock.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert all(line.count("==") == 1 for line in lock_lines)
    locked = {requirement_name(line) for line in lock_lines}
    assert declared <= locked


def test_environment_manifest_matches_lock_and_python_version():
    lines = Path("environment.txt").read_text().splitlines()
    assert lines[0] == "Python 3.14.4"
    manifest = {name.lower().replace("_", "-"): version for name, version in (line.split() for line in lines[1:])}
    locked = dict(line.split("==", maxsplit=1) for line in Path("requirements-lock.txt").read_text().splitlines())
    assert manifest == locked
