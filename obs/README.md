# Preparing openfe-benchmarks systems for SOMD2

Scripts to turn any system in `openfe_benchmarks/data/benchmark_systems` into
equilibrated inputs for [SOMD2](https://github.com/OpenBioSim/somd2), using the
same force fields, partial charges and atom mappings as the OpenFE reference
submissions.

## Requirements

An environment with BioSimSpace, Sire, SOMD2, OpenMM, ParmEd, gufe and RDKit,
with this repository installed (`pip install -e .`). The default protein force
fields (`amber/ff14SB.xml`, `amber/tip3p_standard.xml`,
`amber/tip3p_HFE_multivalent.xml`, `amber/phosaa10.xml`) are provided by
`openmmforcefields`.

## Workflow

```
# 1. Parameterise the protein, ligands and cofactors (cached, so optional).
python obs/parameterise.py --system-group jacs_set --system-name tyk2

# 2a. RBFE: merge and solvate each edge of the ligand network.
python obs/prepare_rbfe.py --system-group jacs_set --system-name tyk2 --list-edges
python obs/prepare_rbfe.py --system-group jacs_set --system-name tyk2 --edges ejm_31~ejm_42

# 2b. ABFE: solvate each ligand and decouple it.
python obs/prepare_abfe.py --system-group jacs_set --system-name tyk2 --ligands ejm_31

# 3. Equilibrate each end state.
python obs/equilibrate.py obs_output/jacs_set/tyk2/rbfe/ejm_31~ejm_42/bound.bss --platform cuda
python obs/equilibrate.py obs_output/jacs_set/tyk2/abfe/ejm_31/bound.bss --end-states 0

# 4. Run SOMD2, seeding the windows closest to lambda = 1 from the perturbed end state.
somd2 bound_lambda_0.bss --perturbed-system bound_lambda_1.bss --replica-exchange ...
```

Without `--edges` or `--ligands`, every edge or ligand in the system is
prepared. Existing outputs are skipped unless `--overwrite` is passed.

## What each step does

- **Parameterise.** The protein, crystal waters and ions are parameterised with
  OpenMM, which matches residue templates by bonding rather than atom name, as
  OpenFE does, and are written to AMBER format with ParmEd. Ligands and
  cofactors are parameterised with OpenFF (`openff_unconstrained-2.3.0` by
  default), then their charges are replaced by the ones in the SDF file
  (`--charge-type`, NAGL by default).
- **Merge.** Edges use the Kartograf mappings stored in the network file. The
  SDF ligands are checked against the network components atom by atom.
  `--allow-ring-breaking` and `--allow-ring-size-change` are passed to
  `BSS.Align.merge` for ring-breaking edges.
- **Solvate.** TIP3P with 0.15 M NaCl, neutralised. The box width follows the
  `openmm.app.Modeller` padding rule, with 1.0 nm padding for the bound leg and
  1.5 nm for the free leg, as in the OpenFE reference submissions.
- **Equilibrate.** Each end state is linked with `sr.morph.link_to_reference`
  or `sr.morph.link_to_perturbed` and run through the staged protocol in
  `config/equilibration.yaml` at the matching λ. The output for end state *n*
  holds the equilibrated coordinates in `coordinates<n>` of the perturbable
  molecule, alongside the equilibrated solvent and box.

## Output layout

```
obs_output/<system_group>/<system_name>/
  parameterised/            protein.prm7/rst7, ligands/*.bss, cofactors.bss
  rbfe/<ligA>~<ligB>/       bound.bss, free.bss, edge.json, *_lambda_{0,1}.bss
  abfe/<ligand>/            bound.bss, free.bss, ligand.json, *_lambda_0.bss
```

`edge.json` records the mapping, the net charge of each end state, and the
solvation settings.
