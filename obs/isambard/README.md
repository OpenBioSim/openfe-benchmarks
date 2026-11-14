# Running the benchmarks on Isambard-AI

Scripts to run SOMD2 on the equilibrated inputs from `obs/equilibrate_all.py`, with
one Slurm job per leg and replicate, then analyse and collect the results. What to
run is set in `config.yaml`, so the set, blocks, systems, replicates and GPU usage
can be changed without touching the scripts.

## Layout

| File | Purpose |
|---|---|
| `config.yaml` | Paths, environment set-up, Slurm settings, and what to run. |
| `somd2/*.yaml` | SOMD2 configs, chosen per block. Settings under `legs:` apply to one leg only. |
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

## Running

From a login node, with the pixi environment active:

```
python benchmarks.py plan              # progress, problems, and GPU-hours still to run
python benchmarks.py submit --dry-run  # what would be submitted
python benchmarks.py submit            # submit up to slurm.max_jobs jobs
python benchmarks.py collect           # results.csv in paths.output
```

Each job submits more legs when it finishes (`top_up: true`), so the queue stays at
`slurm.max_jobs` without anything running on the login node. `max_jobs` therefore
also caps the GPUs in use, at `max_jobs * gpus_per_job`. To pause, set `top_up:
false`, or lower `max_jobs`.

`plan` calls `squeue` once; there's no need to run it more often than every few
minutes. Finished legs are never rerun, so adding blocks, systems or replicates to
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
