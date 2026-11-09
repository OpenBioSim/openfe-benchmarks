"""List the BioSimSpace-mapped RBFE edges whose mapping pairs atoms further apart than
a cut-off in the input poses, i.e. edges prepared before such pairs were dropped, and
print the commands to re-prepare them."""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

import _common
from prepare_rbfe import MAX_PAIR_DISTANCE

logger = _common.logger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="obs_output", help="Root output directory.")
    parser.add_argument("--max-pair-distance", type=float, default=MAX_PAIR_DISTANCE)
    parser.add_argument("--charge-type", default=_common.DEFAULT_CHARGE_TYPE)
    args = parser.parse_args()
    _common.setup_logging(logging.WARNING)

    affected = defaultdict(list)
    poses = {}
    for path in sorted(Path(args.output_dir).glob("*/*/rbfe/*/edge.json")):
        edge = json.loads(path.read_text())
        method = edge.get("mapping_method", {})
        if method.get("mapper") != "biosimspace" or "fallback" in method:
            continue
        if "pruned_pairs" in method:
            continue

        system = (edge["system_group"], edge["system_name"])
        if system not in poses:
            data = _common.get_benchmark(*system, args.charge_type)
            poses[system] = {
                name: mol.GetConformer().GetPositions()
                for name, mol in _common.read_sdf(data.ligands[args.charge_type]).items()
            }
        xa = poses[system][edge["ligand_a"]]
        xb = poses[system][edge["ligand_b"]]

        far = [
            (int(i), j, float(np.linalg.norm(xa[int(i)] - xb[j])))
            for i, j in edge["mapping"].items()
            if np.linalg.norm(xa[int(i)] - xb[j]) > args.max_pair_distance
        ]
        if far:
            name = path.parent.name
            affected[system].append(name)
            pairs = ", ".join(f"{i}-{j} {d:.2f} A" for i, j, d in far)
            print(f"{system[0]}/{system[1]}/{name}: {pairs}")

    print(f"\n{sum(len(v) for v in affected.values())} edges to re-prepare:\n")
    for (group, name), edges in sorted(affected.items()):
        print(
            f"python obs/prepare_rbfe.py --system-group {group} --system-name {name} "
            f"--edges {' '.join(edges)} --overwrite"
        )


if __name__ == "__main__":
    main()
