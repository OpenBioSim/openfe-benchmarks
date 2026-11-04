"""Create solvated bound and free systems with a decoupled ligand for absolute
binding free energy calculations with SOMD2."""

import argparse
import tempfile

import BioSimSpace as BSS
import sire as sr

import _common

logger = _common.logger


def decouple(system):
    mols = sr.system.System(system._sire_object)
    ligand = mols[f"resname {_common.LIGAND_RESNAME}"].molecule()
    mols.update(sr.morph.decouple(ligand, as_new_molecule=False))
    return mols


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    _common.add_common_args(parser)
    _common.add_solvation_args(parser)
    parser.add_argument("--ligands", nargs="+", default=None)
    parser.add_argument("--legs", nargs="+", default=["bound", "free"])
    args = parser.parse_args()

    _common.setup_logging()

    data = _common.get_benchmark(args.system_group, args.system_name, args.charge_type)
    names = list(_common.read_sdf(data.ligands[args.charge_type]))
    if args.ligands is not None:
        missing = set(args.ligands) - set(names)
        if missing:
            raise ValueError(f"Ligands not in the data set: {sorted(missing)}")
        names = args.ligands

    param_dir = _common.parameterise_system(args, data)
    receptor = _common.load_receptor(param_dir) if "bound" in args.legs else None

    root = _common.system_dir(args.output_dir, args.system_group, args.system_name) / "abfe"
    for name in names:
        out = root / _common.safe_name(name)
        if not args.overwrite and all((out / f"{leg}.bss").exists() for leg in args.legs):
            logger.info(f"Skipping {name}: already prepared")
            continue
        out.mkdir(parents=True, exist_ok=True)
        logger.info(f"Preparing {name}")

        ligand = _common.load_ligand(param_dir, name)
        for leg in args.legs:
            if leg == "bound":
                system = receptor.copy()
                system.addMolecules(ligand)
                padding = args.bound_padding
            elif leg == "free":
                system = ligand.toSystem()
                padding = args.free_padding
            else:
                raise ValueError(f"Unknown leg '{leg}'")

            with tempfile.TemporaryDirectory() as tmp:
                system = _common.solvate(
                    system, padding, args.box_shape, args.ion_conc, work_dir=tmp
                )
            sr.stream.save(decouple(system), str(out / f"{leg}.bss"))

        _common.write_json(
            out / "ligand.json",
            {
                "system_group": args.system_group,
                "system_name": args.system_name,
                "ligand": name,
                "charge": _common.net_charge(ligand),
                "box_shape": args.box_shape,
                "bound_padding": args.bound_padding,
                "free_padding": args.free_padding,
                "ion_conc": args.ion_conc,
            },
        )


if __name__ == "__main__":
    main()
