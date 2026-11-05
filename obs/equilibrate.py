"""Equilibrate the end states of a perturbable system with Sire, using a staged
protocol of restrained minimisation and dynamics. Writes one stream file per end
state, which can be used as the input (lambda = 0) and 'perturbed_system'
(lambda = 1) for SOMD2."""

import argparse
import logging
from pathlib import Path

import numpy as np
import sire as sr
import yaml
from sire.legacy.IO import setCoordinates

logger = logging.getLogger("obs")

# The SOMD2 defaults.
CONSTRAINT = "h-bonds"
PERTURBABLE_CONSTRAINT = "h-bonds-not-heavy-perturbed"

# Constraints can fail for perturbable molecules with many ghost atoms. If a
# dynamics stage blows up, run briefly without them and try again.
INSTABILITY_ERRORS = ("NaN", "kinetic energy has exceeded")
FALLBACK_RUNTIME = "10 ps"
FALLBACK_TIMESTEP = "1 fs"


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
        "constraint": stage.get("constraint", CONSTRAINT),
        "perturbable_constraint": stage.get(
            "perturbable_constraint", PERTURBABLE_CONSTRAINT
        ),
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
        try:
            return run_dynamics(mols, stage, options)
        except Exception as e:
            if not any(text in str(e) for text in INSTABILITY_ERRORS):
                raise
            logger.warning(
                f"Dynamics failed ({str(e).splitlines()[0]}). Running "
                f"{stage.get('fallback_runtime', FALLBACK_RUNTIME)} without constraints "
                "on perturbable molecules, then retrying."
            )

        unconstrained = {
            **stage,
            "runtime": stage.get("fallback_runtime", FALLBACK_RUNTIME),
            "timestep": stage.get("fallback_timestep", FALLBACK_TIMESTEP),
        }
        mols = run_dynamics(mols, unconstrained, {**options, "perturbable_constraint": "none"})
        return run_dynamics(mols, stage, options)

    raise ValueError(f"Unknown stage type '{kind}'")


def run_dynamics(mols, stage, options):
    d = mols.dynamics(
        timestep=stage["timestep"],
        temperature=stage["temperature"],
        pressure=stage.get("pressure"),
        **options,
    )
    d.run(stage["runtime"], save_frequency=0, auto_fix_minimise=False)
    return d.commit()


def with_coordinates(system, coords, state, space):
    """Return a copy of 'system' with 'coords' as the coordinates of end state
    'state', keeping those of the other end state."""
    result = sr.system.System(setCoordinates(system._system, coords.tolist()))
    result.set_space(space)

    # setCoordinates writes to 'coordinates0' of perturbable molecules.
    for mol in result.molecules("property is_perturbable"):
        number = f"molnum {mol.number().value()}"
        cursor = mol.cursor()
        cursor[f"coordinates{state}"] = mol.property("coordinates0")
        cursor[f"coordinates{1 - state}"] = system[number].molecule().property(
            f"coordinates{1 - state}"
        )
        result.update(cursor.commit())
    return result


def rescale(system, space):
    """Scale the molecule centres of the lambda = 1 end state of 'system' so that it
    fits in 'space', which has the same shape but a different volume."""
    scale = (space.volume().value() / system.space().volume().value()) ** (1 / 3)
    logger.info(f"Rescaling the lambda = 1 box by {scale:.5f} to match lambda = 0")

    coords = sr.io.get_coords_array(sr.morph.link_to_perturbed(system))
    counts = np.array([mol.num_atoms() for mol in system.molecules()])
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    centres = np.add.reduceat(coords, starts) / counts[:, None]
    coords += (scale - 1) * np.repeat(centres, counts, axis=0)
    return with_coordinates(system, coords, 1, space)


def reduce_box(system):
    """Put a triclinic box in the reduced form OpenMM requires. Inputs from before
    this was done in solvation can be marginally outside it because of rounding."""
    space = system.space()
    if hasattr(space, "reduce"):
        space.reduce(0)
        system.set_space(space)


def run_stages(mols, stages, defaults, state, platform, name="stage"):
    for i, stage in enumerate(stages):
        logger.info(f"lambda = {state}: {name} {i + 1}/{len(stages)} {stage}")
        mols = run_stage(mols, stage, defaults, float(state), platform)
    return mols


def equilibrate(system, protocol, state, platform, box=None):
    """Equilibrate end state 'state' of 'system'. If 'box' is given, the result is
    rescaled to it and re-equilibrated at constant volume, so that both end states
    share the same box, as needed for GCMC."""
    link = sr.morph.link_to_reference if state == 0 else sr.morph.link_to_perturbed
    defaults = protocol.get("defaults", {})

    mols = run_stages(link(system), protocol["stages"], defaults, state, platform)
    result = with_coordinates(system, sr.io.get_coords_array(mols), state, mols.space())

    if box is not None:
        stages = [{**stage, "pressure": None} for stage in protocol.get("match_box", [])]
        mols = run_stages(
            link(rescale(result, box)), stages, defaults, state, platform, "match_box stage"
        )
        result = with_coordinates(system, sr.io.get_coords_array(mols), state, mols.space())

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
    parser.add_argument(
        "--match-box",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rescale the lambda = 1 box to the lambda = 0 one, as needed for GCMC.",
    )
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
    reduce_box(system)
    try:
        system.molecules("property is_perturbable")
    except KeyError:
        raise ValueError(f"No perturbable molecules in {input_file}")

    for state in sorted(args.end_states):
        if state not in (0, 1):
            raise ValueError("End states must be 0 or 1")
        output = output_dir / f"{input_file.stem}_lambda_{state}.bss"
        if output.exists() and not args.overwrite:
            logger.info(f"Skipping {output}: already exists")
            continue

        box = None
        if state == 1 and args.match_box:
            reference = output_dir / f"{input_file.stem}_lambda_0.bss"
            if not reference.exists():
                raise ValueError(
                    f"{reference} is needed to match the lambda = 1 box. Equilibrate "
                    "lambda = 0 first, or pass --no-match-box."
                )
            box = sr.stream.load(str(reference)).space()

        result = equilibrate(system, protocols[leg], state, args.platform, box)
        sr.stream.save(result, str(output))
        logger.info(f"Saved {output}")


if __name__ == "__main__":
    main()
