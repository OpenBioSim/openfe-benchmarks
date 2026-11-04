"""Translate hand-curated atom mappings from a scaffold-hopping-fep-benchmark
network onto the ligands of an openfe-benchmarks system, so the edges can be
prepared with prepare_rbfe.py --mappings. Ligands are matched by connectivity,
and atoms by the graph isomorphism that best overlays the two poses."""

import argparse
import json
from pathlib import Path

import numpy as np
from rdkit import Chem

import _common

MAX_MATCHES = 200000


def kabsch_rmsd(x, y):
    x = x - x.mean(axis=0)
    y = y - y.mean(axis=0)
    u, _, vt = np.linalg.svd(x.T @ y)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return np.sqrt(((x @ r - y) ** 2).sum(axis=1).mean())


def skeleton(mol):
    return Chem.MolToInchiKey(Chem.RemoveHs(mol)).split("-")[0]


def atom_order(source, target):
    """Return {source index: target index} for the same molecule in two files."""
    if source.GetNumAtoms() != target.GetNumAtoms():
        raise ValueError("Atom counts differ")
    matches = source.GetSubstructMatches(target, uniquify=False, maxMatches=MAX_MATCHES)
    if not matches:
        raise ValueError("No atom correspondence found")
    if len(matches) == MAX_MATCHES:
        raise ValueError("Too many symmetry-equivalent matches")
    xs = source.GetConformer().GetPositions()
    xt = target.GetConformer().GetPositions()
    best = min(matches, key=lambda m: kabsch_rmsd(xs[list(m)], xt))
    return {s: t for t, s in enumerate(best)}, kabsch_rmsd(xs[list(best)], xt)


def broken_bonds(mol_a, mol_b, mapping):
    """Bonds between mapped atoms in A whose images aren't bonded in B."""
    broken = []
    for bond in mol_a.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in mapping and j in mapping and mol_b.GetBondBetweenAtoms(mapping[i], mapping[j]) is None:
            broken.append((i, j))
    return broken


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("network", help="scaffold-hopping-fep-benchmark network JSON")
    parser.add_argument("--edges", nargs="+", required=True, help="edge_id values to translate")
    parser.add_argument("--system-group", required=True)
    parser.add_argument("--system-name", required=True)
    parser.add_argument("--charge-type", default=_common.DEFAULT_CHARGE_TYPE)
    parser.add_argument("--output", required=True, help="Mappings file to add the edges to")
    args = parser.parse_args()
    _common.setup_logging()

    network = Path(args.network)
    root = network.resolve().parent.parent
    entries = {e["edge_id"]: e for e in json.loads(network.read_text())}
    missing = set(args.edges) - set(entries)
    if missing:
        raise ValueError(f"Edges not in {network}: {sorted(missing)}")

    data = _common.get_benchmark(args.system_group, args.system_name, args.charge_type)
    targets = _common.read_sdf(data.ligands[args.charge_type])
    by_skeleton = {}
    for name, mol in targets.items():
        by_skeleton.setdefault(skeleton(mol), []).append(name)

    output = Path(args.output)
    edges = json.loads(output.read_text()) if output.exists() else []
    edges = [e for e in edges if e["source"]["edge_id"] not in args.edges or
             e["source"]["network"] != network.name]

    for edge_id in args.edges:
        entry = entries[edge_id]
        ligands, orders = [], []
        for key in ("ligand_a_paths", "ligand_b_paths"):
            path = root / entry[key][0]
            source = Chem.MolFromMolFile(str(path), removeHs=False)
            names = by_skeleton.get(skeleton(source), [])
            if len(names) != 1:
                raise ValueError(f"{edge_id}: {path.name} matches {names or 'no'} ligands")
            order, rmsd = atom_order(source, targets[names[0]])
            _common.logger.info(f"{edge_id}: {path.name} -> {names[0]} (pose RMSD {rmsd:.2f} A)")
            ligands.append((source, targets[names[0]], names[0]))
            orders.append(order)

        source_mapping = {int(a): int(b) for a, b in entry["mapping"].items()}
        mapping = {orders[0][a]: orders[1][b] for a, b in source_mapping.items()}

        (src_a, tgt_a, name_a), (src_b, tgt_b, name_b) = ligands
        before = broken_bonds(src_a, src_b, source_mapping)
        after = broken_bonds(tgt_a, tgt_b, mapping)
        if sorted(tuple(sorted(orders[0][i] for i in b)) for b in before) != sorted(
            tuple(sorted(b)) for b in after
        ):
            raise ValueError(f"{edge_id}: translated mapping breaks different bonds")
        mixed = sum(
            tgt_a.GetAtomWithIdx(a).GetAtomicNum() != tgt_b.GetAtomWithIdx(b).GetAtomicNum()
            for a, b in mapping.items()
        )
        _common.logger.info(
            f"{edge_id}: {len(mapping)} mapped atoms ({mixed} to a different element), "
            f"{len(after)} bonds broken"
        )

        edges.append(
            {
                "id": f"{_common.safe_name(name_a)}~{_common.safe_name(name_b)}_rb",
                "system": f"{args.system_group}/{args.system_name}",
                "ligand_a": name_a,
                "ligand_b": name_b,
                "mapping": {str(a): b for a, b in sorted(mapping.items())},
                "allow_ring_breaking": True,
                "allow_ring_size_change": True,
                "metadata": entry["metadata"],
                "source": {"network": network.name, "edge_id": edge_id},
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    _common.write_json(output, edges)
    _common.logger.info(f"Wrote {len(edges)} edges to {output}")


if __name__ == "__main__":
    main()
