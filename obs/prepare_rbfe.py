"""Create solvated, merged bound and free systems for the edges of an
openfe-benchmarks ligand network, using the atom mappings stored in the network."""

import argparse
import tempfile

import BioSimSpace as BSS

import _common

logger = _common.logger


def merge_edge(edge, param_dir, allow_ring_breaking, allow_ring_size_change):
    lig_a = _common.load_ligand(param_dir, edge.componentA.name)
    lig_b = _common.load_ligand(param_dir, edge.componentB.name)

    mapping = {int(a): int(b) for a, b in edge.componentA_to_componentB.items()}
    inverse = {b: a for a, b in mapping.items()}

    lig_b = BSS.Align.rmsdAlign(lig_b, lig_a, inverse)
    merged = BSS.Align.merge(
        lig_a,
        lig_b,
        mapping,
        allow_ring_breaking=allow_ring_breaking,
        allow_ring_size_change=allow_ring_size_change,
    )
    return merged, lig_a, lig_b, mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    _common.add_common_args(parser)
    _common.add_solvation_args(parser)
    parser.add_argument("--network", default=None, help="Network file stem.")
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
    for node in network.nodes:
        _common.check_same_molecule(sdf[node.name], node)

    param_dir = _common.parameterise_system(args, data)
    receptor = _common.load_receptor(param_dir) if "bound" in args.legs else None

    root = _common.system_dir(args.output_dir, args.system_group, args.system_name) / "rbfe"
    for edge in edges:
        name = _common.edge_name(edge)
        out = root / name
        if not args.overwrite and all((out / f"{leg}.bss").exists() for leg in args.legs):
            logger.info(f"Skipping {name}: already prepared")
            continue
        out.mkdir(parents=True, exist_ok=True)
        logger.info(f"Preparing {name}")

        merged, lig_a, lig_b, mapping = merge_edge(
            edge, param_dir, args.allow_ring_breaking, args.allow_ring_size_change
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
