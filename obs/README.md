# Preparing openfe-benchmarks systems for SOMD2

Scripts to turn any system in `openfe_benchmarks/data/benchmark_systems` into
equilibrated inputs for [SOMD2](https://github.com/OpenBioSim/somd2), using the
same force fields, partial charges and atom mappings as the OpenFE reference
submissions. Membrane systems (`gpcrs`) aren't supported yet, since solvation
uses a water box.

## Installation

The scripts need the OpenBioSim stack (SOMD2, BioSimSpace and Sire) plus:

- `openmmforcefields`, which provides the OpenFE default protein force fields
  (`amber/ff14SB.xml`, `amber/tip3p_standard.xml`,
  `amber/tip3p_HFE_multivalent.xml`, `amber/phosaa10.xml`);
- `parmed`, `gufe`, `rdkit` and `pyyaml`;
- this repository, for the benchmark data.

### Conda

```
conda create -n obs -c conda-forge -c openbiosim \
    somd2 openmmforcefields parmed gufe rdkit pyyaml
conda activate obs
pip install --no-deps -e /path/to/openfe-benchmarks
```

Use `-c openbiosim/label/dev` for the development versions of the OpenBioSim
packages.

### Existing pixi environment

From a Sire, BioSimSpace or SOMD2 pixi environment, install the extra packages
with pip, which leaves `pixi.toml` and `pixi.lock` unchanged:

```
pixi shell
pip install openmmforcefields
pip install -e /path/to/openfe-benchmarks
```

### GPU platform

If OpenMM has no CUDA platform, pass `--platform opencl` to `equilibrate.py`,
and set `OPENCL_VISIBLE_DEVICES` to the GPU to use when running SOMD2.

## Workflow

Run from the root of this repository.

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
prepared. Existing outputs are skipped unless `--overwrite` is passed, so
interrupted runs can be restarted. Steps 1 and 2 take a few seconds per ligand
or edge on a single CPU core.

### Benchmark sets

The curated set for the SOMD2 paper is defined in `config/curated.yaml`, as
blocks of RBFE and ABFE systems. To prepare and equilibrate it, or some of its
blocks:

```
python obs/prepare_set.py --set curated --jobs 8
python obs/equilibrate_all.py --set curated --gpus 0 1 2
python obs/equilibrate_all.py --set curated --blocks A G --gpus 0 1 2
```

A block can take its edges from a mappings file in `config/mappings/` rather
than the OpenFE network. `ring_breaking.json` holds ring-breaking edges from the
scaffold-hopping-fep-benchmark repository, with their hand-curated mappings
translated onto the OpenFE ligands by `translate_mappings.py`:

```
python obs/translate_mappings.py path/to/networks/zou_network.json \
    --edges chk1_c20_to_c17 --system-group scaffold_hopping --system-name chk1 \
    --output obs/config/mappings/ring_breaking.json
```

Ligands are matched by connectivity, and atoms by the isomorphism that best
overlays the two poses. The script checks that the translated mapping breaks the
same bonds as the original. Edges from a mappings file are prepared with
`prepare_rbfe.py --mappings FILE`, and written to `rbfe/<ligA>~<ligB>_rb/`.

`prepare_set.py` runs `--jobs` systems at once, logging each to
`logs/<group>_<system>.log`, and passes any options it doesn't recognise (e.g.
`--output-dir`, `--overwrite`) on to the preparation scripts. Ring-breaking
options set on a block are passed to `prepare_rbfe.py` for its systems.

## Usage

Run any script with `--help` for the full list of options.

### Common options

`parameterise.py`, `prepare_rbfe.py` and `prepare_abfe.py` share these:

| Option | Default | Description |
|---|---|---|
| `--system-group` | required | Benchmark group, e.g. `jacs_set`. |
| `--system-name` | required | System within the group, e.g. `tyk2`. |
| `--output-dir` | `obs_output` | Root output directory. |
| `--charge-type` | `nagl_openff-gnn-am1bcc-1.0.0.pt` | Partial charges to read from the SDF files. |
| `--ligand-forcefield` | `openff_unconstrained-2.3.0` | OpenFF force field for ligands and cofactors. |
| `--protein-forcefields` | OpenFE defaults | OpenMM force field XML files for the protein, crystal waters and ions. |
| `--overwrite` | off | Regenerate existing outputs. |

Parameters are cached in `parameterised/`. If they were generated with a
different charge type or force field, the scripts stop rather than mix them;
pass `--overwrite` to regenerate them.

### `prepare_rbfe.py`

| Option | Default | Description |
|---|---|---|
| `--network` | `industry_benchmarks_network`, else `lomap_network` | Ligand network file stem. |
| `--mapper` | `biosimspace` | Map atoms with BioSimSpace, or use the network's mappings (`openfe`). |
| `--mappings` | | Take the edges and mappings from a mappings file instead of the network. |
| `--list-edges` | | Print the edges in the network and exit. |
| `--edges` | all | Edges to prepare, as `ligA~ligB`. |
| `--legs` | `bound free` | Legs to prepare. |
| `--allow-ring-breaking` | off | Allow merges that break rings. |
| `--allow-ring-size-change` | off | Allow merges that change ring size. |
| `--bound-padding` | `1.0 nm` | Solvent padding for the bound leg. |
| `--free-padding` | `1.5 nm` | Solvent padding for the free leg. |
| `--box-shape` | `dodecahedron` | `cube`, `dodecahedron` or `octahedron`. |
| `--ion-conc` | `0.15` | NaCl concentration in mol/L. The system is also neutralised. |

### `prepare_abfe.py`

Takes `--ligands` (default: all) and `--legs`, plus the solvation options above.

### `equilibrate.py`

```
python obs/equilibrate.py INPUT [options]
```

| Option | Default | Description |
|---|---|---|
| `INPUT` | | A `bound.bss` or `free.bss` file from step 2. |
| `--protocol` | `config/equilibration.yaml` | Staged equilibration protocol. |
| `--leg` | input file stem | Protocol section to use. |
| `--end-states` | `0 1` | End states to equilibrate. ABFE only needs `0`. |
| `--match-box` | on | Rescale the λ = 1 box to the λ = 0 one, as needed for GCMC. Needs `<leg>_lambda_0.bss`. |
| `--platform` | Sire default | OpenMM platform, e.g. `cuda`, `opencl` or `cpu`. |
| `--output-dir` | input directory | Where to write `<leg>_lambda_<n>.bss`. |
| `--overwrite` | off | Regenerate existing outputs. |

### `equilibrate_all.py`

Equilibrates every prepared input under `--output-dir` that doesn't have an
output yet, running one input (both of its end states, in turn) at a time on
each GPU and bound legs first:

```
python obs/equilibrate_all.py --gpus 0 1 2 --systems 'jacs_set/*' 'water_set/*'
```

`--gpus` takes the indices shown by `nvidia-smi`; you don't need to set
`CUDA_VISIBLE_DEVICES` yourself. Each job is pinned to its GPU with
`CUDA_VISIBLE_DEVICES` and `OPENCL_VISIBLE_DEVICES`, and logs to
`<leg>_equilibrate.log` next to its input. `--set` and `--blocks` restrict the jobs
to a benchmark set, `--kinds rbfe` or `--kinds abfe` to one job type, and
`--dry-run` lists the jobs. A failed job doesn't stop the others, and re-running picks up
whatever is left.

### Equilibration protocol

Each stage in the protocol file is a minimisation or dynamics run. A stage can
set `restraint` (`heavy`, `backbone`, or any Sire selection) with force constant
`k`, and overrides the section's `defaults`. The `match_box` stages are run at
constant volume after the λ = 1 box has been rescaled.

## What each step does

- **Parameterise.** The protein, crystal waters and ions are parameterised with
  OpenMM, which matches residue templates by bonding rather than atom name, as
  OpenFE does, and are written to AMBER format with ParmEd. Ligands and
  cofactors are parameterised with OpenFF (`openff_unconstrained-2.3.0` by
  default), then their charges are replaced by the ones in the SDF file
  (`--charge-type`, NAGL by default).
- **Map and merge.** The network decides which ligand pairs are run, but the
  atoms are mapped with `BSS.Align.matchAtoms`, since the OpenFE (Kartograf)
  mappings are designed for OpenFE's hybrid topology and avoid perturbations,
  such as constraint changes, that SOMD2 handles. If BioSimSpace warns that
  relaxing `ringMatchesRingOnly` gives a larger common core, the edge is
  re-mapped with `mcs_kwargs={"ringMatchesRingOnly": False}`. `--mapper openfe`
  uses the network mappings instead, and edges from a mappings file always use
  their own. `edge.json` records which was used. The SDF ligands are checked
  against the network components atom by atom. `--allow-ring-breaking` and
  `--allow-ring-size-change` are passed to `BSS.Align.merge` for ring-breaking
  edges. An edge prepared with a different mapper or `--max-scoring-matches` is
  re-prepared, and its old equilibrated outputs are removed. If the BioSimSpace
  mapping can't be merged, e.g. because it maps across rings of different sizes,
  the OpenFE mapping is used instead and recorded as a fallback in `edge.json`.
  Use a BioSimSpace version with the fix for issue #561, since scoring the
  default 1000 candidate matches could otherwise take minutes per edge.
- **Solvate.** TIP3P with 0.15 M NaCl, neutralised. The box width follows the
  `openmm.app.Modeller` padding rule, with 1.0 nm padding for the bound leg and
  1.5 nm for the free leg, as in the OpenFE reference submissions.
- **Equilibrate.** Each end state is linked with `sr.morph.link_to_reference`
  or `sr.morph.link_to_perturbed` and run through the staged protocol in
  `config/equilibration.yaml` at the matching λ. The output for end state *n*
  is the whole perturbable system, with the equilibrated coordinates in
  `coordinates<n>` of the perturbable molecule, alongside the equilibrated
  solvent and box. The other end state's coordinates are left as prepared.
  Dynamics uses SOMD2's default constraints (`h-bonds`, and
  `h-bonds-not-heavy-perturbed` for perturbable molecules). If a dynamics
  stage blows up, which can happen when constraints meet many ghost atoms, it
  is restarted with 10 ps at 1 fs without constraints on perturbable
  molecules, then retried with constraints (`fallback_runtime` and
  `fallback_timestep` in the protocol override these).
  GCMC needs constant volume, so both end states must share a box: after the
  λ = 1 run, molecule centres are scaled to fit the λ = 0 box, then the system
  is minimised and run for 50 ps at constant volume.

## Output layout

```
obs_output/<system_group>/<system_name>/
  parameterised/            protein.prm7/rst7, ligands/*.bss, cofactors.bss, metadata.json
  rbfe/<ligA>~<ligB>/       bound.bss, free.bss, edge.json, *_lambda_{0,1}.bss
  abfe/<ligand>/            bound.bss, free.bss, ligand.json, *_lambda_0.bss
```

`edge.json` records the mapping, the net charge of each end state and the
solvation settings; `ligand.json` records the same for ABFE. These files and
`metadata.json` also record provenance: the openfe-benchmarks commit, whether
its data had local changes, and the versions of the packages used.
