# Running the benchmarks on Isambard-AI

Scripts to run SOMD2 on the equilibrated inputs from `obs/equilibrate_all.py`, with
one Slurm job per leg and replicate, then analyse and collect the results. What to
run is set in `config.yaml`, so the set, blocks, systems, replicates and GPU usage
can be changed without touching the scripts.

## Layout

| File | Purpose |
|---|---|
| `config.yaml` | Paths, environment set-up, Slurm settings, and what to run. |
| `config_calibration.yaml` | Calibration of the default SOMD2 settings, with `config_calibration_smoke.yaml` as a short check of it first. `config_rest2.yaml` calibrates the REST2 scale on the same edges. |
| `config_debug.yaml` | Reruns of unstable legs with crash reports, written to each leg's run directory, e.g. with terminal flips or GCMC switched off, or the co-alchemical ion restrained. |
| `somd2/*.yaml` | SOMD2 configs, chosen per block. Settings under `legs:` apply to one leg only, and `base:` names a config to start from. |
| `benchmarks.py` | Run on a login node: `plan`, `submit` and `collect`. |
| `leg.sbatch`, `leg.py` | One job: run SOMD2, analyse with MBAR, copy the results, top up the queue. |

SOMD2 runs in `paths.scratch`, so checkpoints for restarts stay there. When a leg
finishes, its energies, config, log and `analysis.json` are copied to `paths.output`,
under `<group>/<system>/<rbfe|abfe>/<name>/<config>/<leg>/rep<n>/`.

## Setting up

1. Copy this repository (the `feature_obs` branch) to Isambard-AI, e.g. to
   `$PROJECTDIR/lester/code/openfe-benchmarks`, and the equilibrated inputs to
   `paths.input`.
2. Check `config.yaml`: the paths, `setup` (CUDA and GCC modules, and the pixi
   environment), `blocks`, `replicates`, and `slurm.max_jobs`. GCMC compiles CUDA
   kernels with nvcc, which doesn't support pixi's GCC, so `setup` points it at
   the system GCC 13.
3. Fill in the TODOs in `somd2/*.yaml` for the configs in use.

A block can be run with several SOMD2 configs by giving a list, e.g. `A: [matched,
all_on]`. Each config's legs are kept separately, under the config's name. Settings
under `overrides:` in `config.yaml` are applied to the SOMD2 config of every leg,
e.g. a short `runtime` for a test. A test with overrides should use its own
`paths.scratch` and `paths.output`, so that its legs don't count as done for the
real run. A top-level `legs:` list, e.g. `[free]`, runs only those legs. An `exclude:` list of systems or edges, in the same form as `only:`, leaves them out.

The throughput used for GPU-hour estimates can be set per SOMD2 config under
`throughput.configs`. A config without a value uses the value of the config it's
based on, and then the defaults.

## Running

From a login node, with the pixi environment active:

```
python benchmarks.py plan              # progress, problems, and GPU-hours still to run
python benchmarks.py progress          # how far each leg that isn't done has got
python benchmarks.py submit --dry-run  # what would be submitted
python benchmarks.py submit            # submit up to slurm.max_jobs jobs
python benchmarks.py collect           # results.csv in paths.output
```

Each job submits more legs when it finishes (`top_up: true`), so the queue stays at
`slurm.max_jobs` without anything running on the login node. `max_jobs` therefore
also caps the GPUs in use, at `max_jobs * gpus_per_job`. To pause, set `top_up:
false`, or lower `max_jobs`.

`plan` calls `squeue` once; there's no need to run it more often than every few
minutes. `progress` reads each running leg's SOMD2 log in `paths.scratch`, and shows
the replica exchange cycles run and an estimate of the time left. The total
number of cycles is known once the leg has reached its first checkpoint. Finished legs are never rerun, so adding blocks, systems or replicates to
`config.yaml` and running `submit` again only submits the new work.

## Failures and restarts

A leg whose job ended without finishing (time limit, node failure, crash) is shown
as `unfinished` and resubmitted by the next `submit`. SOMD2 restarts it from its last
checkpoint. After `slurm.max_attempts` attempts it's shown as `failed`, with its
latest Slurm log. To retry a failed leg, delete the `attempt.*` files in its scratch
directory.

A leg that becomes unstable (a `NaN` or SOMD2 reporting it unstable) is restarted
from scratch with `fallback_timestep`, since SOMD2 can't change the timestep on a
restart. The failed run is kept in `somd2.unstable.<timestep>` in the scratch
directory, and later attempts keep the fallback timestep. Each `analysis.json`
records the timestep used.

## Results

`collect` writes one row per edge or ABFE ligand: the mean over replicates, its
standard error, the mean MBAR error, the smallest neighbouring-window overlap, and
the value for each replicate. RBFE gives ΔΔG = ΔG(bound) − ΔG(free). ABFE gives
ΔG = ΔG(free) − ΔG(bound) − the Boresch standard-state correction, which SOMD2
records in the energy files. Each `analysis.json` also has the PMF, the overlap
matrix and a forward/backward convergence check.
