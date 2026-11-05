"""Create solvated, merged bound and free systems for the edges of an
openfe-benchmarks ligand network. Atoms are mapped with BioSimSpace by default,
or taken from the network ('--mapper openfe') or a mappings file ('--mappings',
see translate_mappings.py)."""

import argparse
import json
import tempfile
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import BioSimSpace as BSS

import _common

logger = _common.logger


# BioSimSpace warns with this text when relaxing ring matching gives a larger MCS.
RELAXED_RING_WARNING = "ringMatchesRingOnly"


def bss_mapping(lig_a, lig_b, max_scoring_matches):
    """Map with BioSimSpace, re-mapping with ring matching relaxed if BioSimSpace
    warns that this gives a larger common core."""
    start = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mapping = BSS.Align.matchAtoms(
            lig_a, lig_b, max_scoring_matches=max_scoring_matches
        )
    messages = [str(w.message) for w in caught]

    mcs_kwargs = {}
    if any(RELAXED_RING_WARNING in m for m in messages):
        mcs_kwargs = {"ringMatchesRingOnly": False}
        logger.info(f"Re-mapping with {mcs_kwargs}: {len(mapping)} mapped atoms before")
        mapping = BSS.Align.matchAtoms(
            lig_a, lig_b, max_scoring_matches=max_scoring_matches, mcs_kwargs=mcs_kwargs
        )
    logger.info(f"{len(mapping)} mapped atoms in {time.time() - start:.1f} s")
    return mapping, {
        "mcs_kwargs": mcs_kwargs,
        "max_scoring_matches": max_scoring_matches,
        "warnings": messages,
    }


def mapper(args, edge):
    return "file" if hasattr(edge, "allow_ring_breaking") else args.mapper


def merge(lig_a, lig_b, mapping, allow_ring_breaking, allow_ring_size_change):
    inverse = {b: a for a, b in mapping.items()}
    lig_b = BSS.Align.rmsdAlign(lig_b, lig_a, inverse)
    merged = BSS.Align.merge(
        lig_a,
        lig_b,
        mapping,
        allow_ring_breaking=allow_ring_breaking,
        allow_ring_size_change=allow_ring_size_change,
    )
    return merged, lig_b


def merge_edge(edge, method, param_dir, allow_ring_breaking, allow_ring_size_change, max_matches):
    lig_a = _common.load_ligand(param_dir, edge.componentA.name)
    lig_b = _common.load_ligand(param_dir, edge.componentB.name)
    network_mapping = {int(a): int(b) for a, b in edge.componentA_to_componentB.items()}
    flags = (allow_ring_breaking, allow_ring_size_change)

    details = {"mapper": method}
    if method != "biosimspace":
        merged, lig_b = merge(lig_a, lig_b, network_mapping, *flags)
        return merged, lig_a, lig_b, network_mapping, details

    mapping, extra = bss_mapping(lig_a, lig_b, max_matches)
    details.update(extra)
    try:
        merged, lig_b = merge(lig_a, lig_b, mapping, *flags)
    except BSS._Exceptions.IncompatibleError as e:
        # E.g. an MCS across rings of different sizes, which Kartograf filters out.
        logger.warning(f"BioSimSpace mapping can't be merged ({e}). Using the OpenFE mapping.")
        details["fallback"] = {"mapper": "openfe", "reason": str(e)}
        mapping = network_mapping
        merged, lig_b = merge(lig_a, lig_b, mapping, *flags)
    return merged, lig_a, lig_b, mapping, details


def load_mappings(path, system):
    """Return the edges of a mappings file for a system, shaped like gufe edges."""
    edges = []
    for entry in json.loads(Path(path).read_text()):
        if entry["system"] != system:
            continue
        edges.append(
            SimpleNamespace(
                name=entry["id"],
                componentA=SimpleNamespace(name=entry["ligand_a"]),
                componentB=SimpleNamespace(name=entry["ligand_b"]),
                componentA_to_componentB=entry["mapping"],
                annotations={"metadata": entry["metadata"], "source": entry["source"]},
                allow_ring_breaking=entry.get("allow_ring_breaking", False),
                allow_ring_size_change=entry.get("allow_ring_size_change", False),
            )
        )
    if not edges:
        raise ValueError(f"No edges for {system} in {path}")
    return edges


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    _common.add_common_args(parser)
    _common.add_solvation_args(parser)
    parser.add_argument("--network", default=None, help="Network file stem.")
    parser.add_argument(
        "--mapper",
        default="biosimspace",
        choices=["biosimspace", "openfe"],
        help="Map atoms with BioSimSpace, or use the mappings stored in the network.",
    )
    parser.add_argument(
        "--max-scoring-matches",
        type=int,
        default=1000,
        help="Candidate MCS matches that BioSimSpace scores. Large values are slow "
        "with BioSimSpace versions before the fix for issue #561.",
    )
    parser.add_argument(
        "--mappings", default=None, help="Take the edges from a mappings file instead."
    )
    parser.add_argument(
        "--edges", nargs="+", default=None, help="Edges to prepare, as 'ligA~ligB'."
    )
    parser.add_argument("--list-edges", action="store_true")
    parser.add_argument("--legs", nargs="+", default=["bound", "free"])
    parser.add_argument("--allow-ring-breaking", action="store_true")
    parser.add_argument("--allow-ring-size-change", action="store_true")
    args = parser.parse_args()

    _common.setup_logging()

    data = _common.get_benchmark(args.system_group, args.system_name, args.charge_type)
    if args.mappings is not None:
        network = None
        system = f"{args.system_group}/{args.system_name}"
        edges = sorted(load_mappings(args.mappings, system), key=_common.edge_name)
    else:
        network = _common.load_network(data, args.network)
        edges = sorted(network.edges, key=_common.edge_name)

    if args.list_edges:
        for edge in edges:
            print(_common.edge_name(edge))
        return

    if args.edges is not None:
        names = {_common.edge_name(e): e for e in edges}
        missing = set(args.edges) - set(names)
        if missing:
            raise ValueError(f"Edges not in the network: {sorted(missing)}")
        edges = [names[n] for n in args.edges]

    sdf = _common.read_sdf(data.ligands[args.charge_type])
    if network is not None:
        for node in network.nodes:
            _common.check_same_molecule(sdf[node.name], node)
    else:
        for edge in edges:
            for name in (edge.componentA.name, edge.componentB.name):
                if name not in sdf:
                    raise ValueError(f"Ligand '{name}' of edge {edge.name} is not in {system}")

    param_dir = _common.parameterise_system(args, data)
    receptor = _common.load_receptor(param_dir) if "bound" in args.legs else None

    root = _common.system_dir(args.output_dir, args.system_group, args.system_name) / "rbfe"
    failed = []
    for edge in edges:
        name = _common.edge_name(edge)
        out = root / name
        if not args.overwrite and is_prepared(
            out, args.legs, mapper(args, edge), args.max_scoring_matches
        ):
            logger.info(f"Skipping {name}: already prepared")
            continue
        out.mkdir(parents=True, exist_ok=True)
        (out / "edge.json").unlink(missing_ok=True)
        for leg in args.legs:
            for stale in out.glob(f"{leg}_lambda_*"):
                stale.unlink()
        logger.info(f"Preparing {name}")
        try:
            prepare_edge(args, edge, out, param_dir, receptor)
        except Exception:
            logger.exception(f"Failed to prepare {name}")
            failed.append(name)

    _common.report_failures(failed, len(edges), "edges")


def is_prepared(out, legs, method, max_scoring_matches):
    """Whether the edge has been prepared with the same mapping settings."""
    info = out / "edge.json"
    if not info.exists() or not all((out / f"{leg}.bss").exists() for leg in legs):
        return False
    details = json.loads(info.read_text()).get("mapping_method", {})
    if details.get("mapper") != method:
        return False
    if method == "biosimspace":
        return details.get("max_scoring_matches") == max_scoring_matches
    return True


def prepare_edge(args, edge, out, param_dir, receptor):
    merged, lig_a, lig_b, mapping, details = merge_edge(
        edge,
        mapper(args, edge),
        param_dir,
        args.allow_ring_breaking or getattr(edge, "allow_ring_breaking", False),
        args.allow_ring_size_change or getattr(edge, "allow_ring_size_change", False),
        args.max_scoring_matches,
    )

    for leg in args.legs:
        if leg == "bound":
            system = receptor.copy()
            system.addMolecules(merged)
            padding = args.bound_padding
        elif leg == "free":
            system = merged.toSystem()
            padding = args.free_padding
        else:
            raise ValueError(f"Unknown leg '{leg}'")

        with tempfile.TemporaryDirectory() as tmp:
            system = _common.solvate(
                system, padding, args.box_shape, args.ion_conc, work_dir=tmp
            )
        BSS.Stream.save(system, str(out / leg))

    _common.write_json(
        out / "edge.json",
        {
            "system_group": args.system_group,
            "system_name": args.system_name,
            "ligand_a": edge.componentA.name,
            "ligand_b": edge.componentB.name,
            "charge_a": _common.net_charge(lig_a),
            "charge_b": _common.net_charge(lig_b),
            "mapping": mapping,
            "mapping_method": details,
            "annotations": dict(edge.annotations),
            "box_shape": args.box_shape,
            "bound_padding": args.bound_padding,
            "free_padding": args.free_padding,
            "ion_conc": args.ion_conc,
            "provenance": _common.provenance(),
        },
    )


if __name__ == "__main__":
    main()
