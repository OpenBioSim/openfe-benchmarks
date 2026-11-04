"""Load benchmark sets, e.g. config/curated.yaml."""

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent / "config"
KINDS = ("rbfe", "abfe")
MERGE_OPTIONS = ("allow_ring_breaking", "allow_ring_size_change")


def set_file(name):
    path = Path(name)
    if path.suffix not in (".yaml", ".yml"):
        path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise ValueError(f"No benchmark set file {path}")
    return path


def load_set(name, blocks=None):
    """Return a list of (block, kind, group, system, options) entries."""
    data = yaml.safe_load(set_file(name).read_text())["blocks"]
    if blocks is not None:
        missing = set(blocks) - set(data)
        if missing:
            raise ValueError(f"Blocks not in {name}: {sorted(missing)}. Available: {list(data)}")
        data = {b: data[b] for b in blocks}

    entries = []
    for block, spec in data.items():
        options = {k: bool(spec.get(k, False)) for k in MERGE_OPTIONS}
        for kind in KINDS:
            for system in spec.get(kind, []):
                group, _, name = system.partition("/")
                if not name:
                    raise ValueError(f"System '{system}' in block {block} isn't 'group/name'")
                entries.append((block, kind, group, name, options if kind == "rbfe" else {}))
    return entries
