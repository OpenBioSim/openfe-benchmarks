"""Shared helpers for preparing openfe-benchmarks systems for SOMD2."""

import datetime
import importlib
import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path

import BioSimSpace as BSS
import gufe
import numpy as np
import sire as sr
from rdkit import Chem

from openfe_benchmarks.data import get_data_by_system_name

logger = logging.getLogger("obs")

DEFAULT_CHARGE_TYPE = "nagl_openff-gnn-am1bcc-1.0.0.pt"
DEFAULT_LIGAND_FORCEFIELD = "openff_unconstrained-2.3.0"
# The OpenFE defaults, provided by openmmforcefields.
DEFAULT_PROTEIN_FORCEFIELDS = [
    "amber/ff14SB.xml",
    "amber/tip3p_standard.xml",
    "amber/tip3p_HFE_multivalent.xml",
    "amber/phosaa10.xml",
]
NETWORK_PREFERENCE = ["industry_benchmarks_network", "lomap_network"]
LIGAND_RESNAME = "LIG"


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # openfe_benchmarks attaches its own handler.
    logging.getLogger("openfe_benchmarks").propagate = False


def add_common_args(parser):
    parser.add_argument("--system-group", required=True, help="e.g. jacs_set")
    parser.add_argument("--system-name", required=True, help="e.g. tyk2")
    parser.add_argument(
        "--output-dir", default="obs_output", help="Root output directory."
    )
    parser.add_argument("--charge-type", default=DEFAULT_CHARGE_TYPE)
    parser.add_argument("--ligand-forcefield", default=DEFAULT_LIGAND_FORCEFIELD)
    parser.add_argument(
        "--protein-forcefields",
        nargs="+",
        default=DEFAULT_PROTEIN_FORCEFIELDS,
        help="OpenMM force field XML files used for the protein, crystal waters and ions.",
    )
    parser.add_argument("--overwrite", action="store_true")


def system_dir(output_dir, system_group, system_name):
    return Path(output_dir) / system_group / system_name


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.+-]", "_", name)


def get_benchmark(system_group, system_name, charge_type):
    data = get_data_by_system_name(system_group, system_name)
    if charge_type not in data.ligands:
        raise ValueError(
            f"No '{charge_type}' ligands for {system_group}/{system_name}. "
            f"Available: {list(data.ligands)}"
        )
    return data


def load_network(data, network=None):
    if not data.ligand_networks:
        raise ValueError(f"No ligand network for {data.system_group}/{data.system_name}")
    if network is None:
        network = next((n for n in NETWORK_PREFERENCE if n in data.ligand_networks), None)
        if network is None:
            network = sorted(data.ligand_networks)[0]
    return gufe.LigandNetwork.from_json(file=str(data.ligand_networks[network]))


def edge_name(edge):
    return f"{safe_name(edge.componentA.name)}~{safe_name(edge.componentB.name)}"


def read_sdf(path):
    """Return {name: RDKit molecule} for a multi-molecule SDF, keeping hydrogens."""
    mols = {}
    for mol in Chem.SDMolSupplier(str(path), removeHs=False):
        if mol is None:
            raise ValueError(f"Unable to read a molecule from {path}")
        name = mol.GetProp("_Name")
        if name in mols:
            raise ValueError(f"Duplicate molecule name '{name}' in {path}")
        mols[name] = mol
    return mols


def sdf_charges(rdmol):
    try:
        return [a.GetDoubleProp("PartialCharge") for a in rdmol.GetAtoms()]
    except KeyError:
        raise ValueError(f"Molecule '{rdmol.GetProp('_Name')}' has no partial charges")


def check_same_molecule(rdmol, component):
    """Make sure an SDF molecule matches a network component atom-for-atom."""
    other = component.to_rdkit()
    name = rdmol.GetProp("_Name")
    if rdmol.GetNumAtoms() != other.GetNumAtoms():
        raise ValueError(f"Atom count of '{name}' differs between SDF and network")
    z0 = [a.GetAtomicNum() for a in rdmol.GetAtoms()]
    z1 = [a.GetAtomicNum() for a in other.GetAtoms()]
    if z0 != z1:
        raise ValueError(f"Atom order of '{name}' differs between SDF and network")
    x0 = rdmol.GetConformer().GetPositions()
    x1 = other.GetConformer().GetPositions()
    if not np.allclose(x0, x1, atol=1e-3):
        raise ValueError(f"Coordinates of '{name}' differ between SDF and network")


def parameterise_small_molecule(rdmol, forcefield, resname=None):
    """Parameterise an RDKit molecule with OpenFF, keeping its SDF partial charges."""
    charges = sdf_charges(rdmol)
    name = rdmol.GetProp("_Name")

    with tempfile.TemporaryDirectory() as tmp:
        sdf = Path(tmp) / "mol.sdf"
        with Chem.SDWriter(str(sdf)) as writer:
            writer.write(rdmol)
        mol = BSS.IO.readMolecules(str(sdf))[0]
        mol = BSS.Parameters.parameterise(mol, forcefield, work_dir=tmp).getMolecule()

    elements = [a.element().num_protons() for a in mol._sire_object.atoms()]
    if elements != [a.GetAtomicNum() for a in rdmol.GetAtoms()]:
        raise ValueError(f"Parameterisation changed the atom order of '{name}'")

    cursor = mol._sire_object.cursor()
    for atom, charge in zip(cursor.atoms(), charges):
        atom["charge"] = charge * sr.units.mod_electron
    if resname is not None:
        for residue in cursor.residues():
            residue.name = resname
    mol._sire_object = cursor.molecule().commit()

    total = sum(charges)
    formal = Chem.GetFormalCharge(rdmol)
    if abs(total - formal) > 1e-3:
        raise ValueError(
            f"Charges of '{name}' sum to {total:.4f}, but the formal charge is {formal}"
        )
    return mol


def parameterise_protein(pdb_file, forcefields, prm7, rst7):
    """Parameterise protein, crystal waters and ions with OpenMM, as OpenFE does, then
    write AMBER files via ParmEd so they can be read by BioSimSpace."""
    import parmed
    from openmm import app

    pdb = app.PDBFile(str(pdb_file))
    try:
        forcefield = app.ForceField(*forcefields)
    except ValueError as e:
        raise ValueError(
            f"Unable to load force fields {forcefields}. The OpenFE defaults need "
            f"'openmmforcefields' to be installed: {e}"
        )
    system = forcefield.createSystem(
        pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False
    )
    structure = parmed.openmm.load_topology(pdb.topology, system, xyz=pdb.positions)
    structure.save(str(prm7), format="amber", overwrite=True)
    structure.save(str(rst7), format="rst7", overwrite=True)


def parameterise_system(args, data=None):
    """Parameterise the protein, ligands and cofactors of a benchmark system, caching
    the results. Returns the cache directory."""
    if data is None:
        data = get_benchmark(args.system_group, args.system_name, args.charge_type)
    out = system_dir(args.output_dir, args.system_group, args.system_name) / "parameterised"
    (out / "ligands").mkdir(parents=True, exist_ok=True)

    meta_file = out / "metadata.json"
    metadata = {
        "system_group": args.system_group,
        "system_name": args.system_name,
        "charge_type": args.charge_type,
        "ligand_forcefield": args.ligand_forcefield,
        "protein_forcefields": args.protein_forcefields,
    }
    if meta_file.exists() and not args.overwrite:
        previous = json.loads(meta_file.read_text())
        previous.pop("provenance", None)
        if previous != metadata:
            raise ValueError(
                f"Existing parameters in {out} were generated with different settings "
                f"({previous}). Use --overwrite to regenerate them."
            )

    if data.protein is not None:
        prm7, rst7 = out / "protein.prm7", out / "protein.rst7"
        if args.overwrite or not prm7.exists():
            logger.info(f"Parameterising protein {data.protein}")
            parameterise_protein(data.protein, args.protein_forcefields, prm7, rst7)

    for name, rdmol in read_sdf(data.ligands[args.charge_type]).items():
        path = out / "ligands" / f"{safe_name(name)}.bss"
        if args.overwrite or not path.exists():
            logger.info(f"Parameterising ligand {name}")
            mol = parameterise_small_molecule(rdmol, args.ligand_forcefield, LIGAND_RESNAME)
            BSS.Stream.save(mol.toSystem(), str(path.with_suffix("")))

    if data.cofactors is not None:
        if args.charge_type not in data.cofactors:
            raise ValueError(f"No '{args.charge_type}' cofactors for {data.system_name}")
        path = out / "cofactors.bss"
        if args.overwrite or not path.exists():
            cofactors = BSS._SireWrappers.System()
            for name, rdmol in read_sdf(data.cofactors[args.charge_type]).items():
                logger.info(f"Parameterising cofactor {name}")
                cofactors.addMolecules(
                    parameterise_small_molecule(rdmol, args.ligand_forcefield)
                )
            BSS.Stream.save(cofactors, str(path.with_suffix("")))

    write_json(meta_file, {**metadata, "provenance": provenance()})
    return out


def load_ligand(param_dir, name):
    return BSS.Stream.load(str(param_dir / "ligands" / f"{safe_name(name)}.bss"))[0]


def load_receptor(param_dir):
    """Return the protein (with crystal waters and ions) and cofactors as one system."""
    prm7 = param_dir / "protein.prm7"
    if not prm7.exists():
        raise ValueError(f"No parameterised protein in {param_dir}")
    system = BSS.IO.readMolecules([str(prm7), str(param_dir / "protein.rst7")])
    cofactors = param_dir / "cofactors.bss"
    if cofactors.exists():
        system.addMolecules(BSS.Stream.load(str(cofactors)).getMolecules())
    return system


def has_perturbable(mols):
    try:
        return len(mols.molecules("property is_perturbable")) > 0
    except KeyError:
        return False


def box_width(system, padding):
    """Box width from padding, using the same rule as openmm.app.Modeller."""
    mols = sr.system.System(system._sire_object)
    if has_perturbable(mols):
        mols = sr.morph.link_to_reference(mols)
    coords = sr.io.get_coords_array(mols)
    centre = 0.5 * (coords.min(axis=0) + coords.max(axis=0))
    radius = np.linalg.norm(coords - centre, axis=1).max()
    padding = BSS.Types.Length(padding).angstroms().value()
    return max(2 * radius + padding, 2 * padding)


def solvate(system, padding, box_shape, ion_conc, work_dir=None):
    width = BSS.Types.Length(box_width(system, padding), "angstrom")
    if box_shape == "cube":
        box, angles = BSS.Box.cubic(width)
    elif box_shape == "dodecahedron":
        box, angles = BSS.Box.rhombicDodecahedronSquare(width)
    elif box_shape == "octahedron":
        box, angles = BSS.Box.truncatedOctahedron(width)
    else:
        raise ValueError(f"Unsupported box shape '{box_shape}'")
    logger.info(f"Solvating in a {box_shape} box with image distance {width}")
    return BSS.Solvent.tip3p(
        molecule=system,
        box=box,
        angles=angles,
        ion_conc=ion_conc,
        is_neutral=True,
        work_dir=work_dir,
    )


def add_solvation_args(parser):
    # OpenFE (openfe 1.8 with adaptive settings) used 1.0 nm for the complex and
    # 1.5 nm for the solvent leg.
    parser.add_argument("--bound-padding", default="1.0 nm")
    parser.add_argument("--free-padding", default="1.5 nm")
    parser.add_argument(
        "--box-shape", default="dodecahedron", choices=["cube", "dodecahedron", "octahedron"]
    )
    parser.add_argument("--ion-conc", type=float, default=0.15, help="NaCl, mol/L")


def net_charge(mol):
    return round(mol.charge().value())


PROVENANCE_MODULES = [
    "BioSimSpace",
    "sire",
    "openmm",
    "openmmforcefields",
    "parmed",
    "openff.toolkit",
    "openforcefields",
    "gufe",
    "rdkit",
]


def benchmark_commit():
    """Return the openfe-benchmarks commit, and whether its data has local changes."""
    import openfe_benchmarks

    repo = Path(openfe_benchmarks.__file__).resolve().parent.parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", "openfe_benchmarks/data"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"version": openfe_benchmarks.__version__, "commit": None, "dirty": None}
    return {
        "version": openfe_benchmarks.__version__,
        "commit": commit,
        "dirty": bool(status),
    }


def provenance():
    versions = {}
    for name in PROVENANCE_MODULES:
        try:
            versions[name] = getattr(importlib.import_module(name), "__version__", None)
        except ImportError:
            versions[name] = None
    versions["sire"] += f"+{sr.__branch__}.{sr.__revisionid__}"
    return {
        "openfe_benchmarks": benchmark_commit(),
        "versions": versions,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2))

