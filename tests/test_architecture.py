"""Enforce CLAUDE.md architecture rules mechanically, so they can't rot silently."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_games_never_import_core_or_agents():
    offenders = []
    for path in (ROOT / "games").rglob("*.py"):
        for mod in imported_modules(path):
            if mod.split(".")[0] in ("core", "agents"):
                offenders.append(f"{path.relative_to(ROOT)} imports {mod}")
    assert not offenders, offenders


def test_no_global_random_module_in_game_code():
    # Rule 3: all randomness comes from an explicit numpy Generator.
    offenders = []
    for path in (ROOT / "games").rglob("*.py"):
        mods = imported_modules(path)
        if "random" in mods:
            offenders.append(str(path.relative_to(ROOT)))
        text = path.read_text(encoding="utf-8")
        if "np.random.seed(" in text or "np.random.randint(" in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, offenders
