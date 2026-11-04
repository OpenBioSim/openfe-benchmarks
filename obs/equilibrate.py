"""Equilibrate the end states of a perturbable system with Sire, using a staged
protocol of restrained minimisation and dynamics. Writes one stream file per end
state, which can be used as the input (lambda = 0) and 'perturbed_system'
(lambda = 1) for SOMD2."""

import argparse
import logging
from pathlib import Path

import sire as sr
import yaml
from sire.legacy.IO import setCoordinates

logger = logging.getLogger("obs")


def restraint_atoms(mols, kind):
    if kind == "heavy":
        solutes = [m for m in mols.molecules("not water") if m.num_atoms() > 1]
        selection = "not element H"
    elif kind == "backbone":
        solutes = [m for m in mols.molecules("protein")]
        selection = "atomname N, CA, C, O"
    else:
        return mols[kind]

    if not solutes:
        raise ValueError(f"No atoms found for '{kind}' restraints")
    atoms = solutes[0].atoms(selection)
    for mol in solutes[1:]:
        atoms += mol.atoms(selection)
    return atoms


def run_stage(mols, stage, defaults, lambda_value, platform):
    stage = {**defaults, **stage}
    kind = stage.get("type")

    restraints = None
    if stage.get("restraint") is not None:
        restraints = sr.restraints.positional(
            mols, atoms=restraint_atoms(mols, stage["restraint"]), k=stage["k"]
        )

    options = {
        "lambda_value": lambda_value,
        "restraints": restraints,
        "constraint": stage.get("constraint", "h-bonds"),
        "perturbable_constraint": stage.get("perturbable_constraint", "h-bonds"),
        "cutoff": stage.get("cutoff", "9 A"),
        "cutoff_type": "PME",
    }
    if platform is not None:
        options["platform"] = platform

    if kind == "minimisation":
        options["constraint"] = stage.get("constraint", "none")
        options["perturbable_constraint"] = stage.get("perturbable_constraint", "none")
        m = mols.minimisation(**options)
        m.run(max_iterations=stage.get("max_iterations", 10000))
        return m.commit()

    if kind == "dynamics":
        d = mols.dynamics(
            timestep=stage["timestep"],
            temperature=stage["temperature"],
            pressure=stage.get("pressure"),
            **options,
        )
        d.run(stage["runtime"], save_frequency=0, auto_fix_minimise=False)
        return d.commit()

    raise ValueError(f"Unknown stage type '{kind}'")


def equilibrate(system, protocol, state, platform):
    if state == 0:
        mols = sr.morph.link_to_reference(system)
    else:
        mols = sr.morph.link_to_perturbed(system)

    defaults = protocol.get("defaults", {})
    for i, stage in enumerate(protocol["stages"]):
        logger.info(f"lambda = {state}: stage {i + 1}/{len(protocol['stages'])} {stage}")
        mols = run_stage(mols, stage, defaults, float(state), platform)

    coords = sr.io.get_coords_array(mols)

    result = sr.system.System(setCoordinates(system._system, coords.tolist()))
    result.set_space(mols.space())

    # setCoordinates writes to 'coordinates0' of perturbable molecules, so set both
    # end states explicitly.
    for mol in result.molecules("property is_perturbable"):
        number = f"molnum {mol.number().value()}"
        cursor = mol.cursor()
        cursor[f"coordinates{state}"] = mols[number].molecule().property("coordinates")
        cursor[f"coordinates{1 - state}"] = system[number].molecule().property(
            f"coordinates{1 - state}"
        )
        result.update(cursor.commit())

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Stream file, e.g. bound.bss")
    parser.add_argument(
        "--protocol",
        default=str(Path(__file__).parent / "config" / "equilibration.yaml"),
    )
    parser.add_argument(
        "--leg",
        default=None,
        help="Protocol section to use. Defaults to the input file stem, e.g. 'bound'.",
    )
    parser.add_argument("--end-states", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--platform", default=None, help="e.g. cuda, opencl, cpu")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
    )

    input_file = Path(args.input)
    leg = args.leg or input_file.stem
    protocols = yaml.safe_load(Path(args.protocol).read_text())
    if leg not in protocols:
        raise ValueError(f"No '{leg}' protocol in {args.protocol}")

    output_dir = Path(args.output_dir) if args.output_dir else input_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    system = sr.stream.load(str(input_file))
    try:
        system.molecules("property is_perturbable")
    except KeyError:
        raise ValueError(f"No perturbable molecules in {input_file}")

    for state in args.end_states:
        if state not in (0, 1):
            raise ValueError("End states must be 0 or 1")
        output = output_dir / f"{input_file.stem}_lambda_{state}.bss"
        if output.exists() and not args.overwrite:
            logger.info(f"Skipping {output}: already exists")
            continue
        result = equilibrate(system, protocols[leg], state, args.platform)
        sr.stream.save(result, str(output))
        logger.info(f"Saved {output}")


if __name__ == "__main__":
    main()
