"""Run one SOMD2 leg, analyse it, copy the results to the output directory, then top
up the queue. Run inside a job by leg.sbatch."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

COPY = ["energy_traj_*.parquet", "energy_components_*.parquet", "config*.yaml", "log.txt", "abfe_restraint.s3"]


def run_somd2(job, run, work):
    command = ["somd2", job["system"], "--config", str(run / "somd2.yaml"), "--output-directory", str(work)]
    if job["perturbed_system"]:
        command += ["--perturbed-system", job["perturbed_system"]]
    # Replica exchange keeps its state in repex_state.pkl, other runs in checkpoint files.
    restart = (work / "repex_state.pkl").exists() or any(work.glob("checkpoint_*.s3"))
    command += ["--restart"] if restart else ["--overwrite"]
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def windows(work):
    import pyarrow.parquet as pq

    result = []
    for path in work.glob("energy_traj_*.parquet"):
        meta = json.loads(pq.read_metadata(path).metadata[b"somd2"])
        result.append((float(meta["lambda"]), meta, path))
    return sorted(result, key=lambda w: w[0])


def convergence(work, num=10):
    """Free energy from increasing fractions of each window's data, taken from the
    start (forward) and the end (backward)."""
    import BioSimSpace as BSS
    from alchemlyb import concat
    from alchemlyb.estimators import MBAR

    data = [
        BSS.FreeEnergy.Relative._somd2_extract(path, T=float(meta["temperature"]), estimator="MBAR")
        for _, meta, path in windows(work)
    ]
    kT = 0.0019872043 * float(windows(work)[0][1]["temperature"])
    result = {"fraction": [i / num for i in range(1, num + 1)]}
    for direction in ("forward", "backward"):
        values = []
        for i in range(1, num + 1):
            sample = [d.iloc[: len(d) * i // num] if direction == "forward" else d.iloc[-(len(d) * i // num):] for d in data]
            values.append(float(MBAR().fit(concat(sample)).delta_f_.iloc[0, -1]) * kT)
        result[direction] = values
    return result


def analyse(work):
    import BioSimSpace as BSS

    pmf, overlap = BSS.FreeEnergy.Relative.analyse(str(work), estimator="MBAR")
    overlap = np.asarray(overlap)
    meta = windows(work)[0][1]
    correction = meta.get("standard_state_correction")
    result = {
        "free_energy": pmf[-1][1].kcal_per_mol().value() - pmf[0][1].kcal_per_mol().value(),
        "error": pmf[-1][2].kcal_per_mol().value(),
        "lambda": [p[0] for p in pmf],
        "pmf": [p[1].kcal_per_mol().value() for p in pmf],
        "min_overlap": float(min(overlap[i, i + 1] for i in range(len(overlap) - 1))),
        "overlap": overlap.tolist(),
        "standard_state_correction": float(correction) if correction is not None else None,
    }
    try:
        result["convergence"] = convergence(work)
    except Exception as e:
        result["convergence"] = {"error": str(e)}
    return result


def copy_results(work, output, trajectories):
    output.mkdir(parents=True, exist_ok=True)
    patterns = COPY + (["traj_*.dcd"] if trajectories else [])
    for pattern in patterns:
        for path in work.glob(pattern):
            shutil.copy2(path, output / path.name)


def main():
    run = Path(sys.argv[1])
    job = json.loads((run / "job.json").read_text())
    work = run / "somd2"
    output = Path(job["output"])

    run_somd2(job, run, work)
    analysis = analyse(work)
    copy_results(work, output, job["copy_trajectories"])
    (output / "analysis.json").write_text(json.dumps({"id": job["id"], **analysis}, indent=2))
    print(f"{job['id']}: {analysis['free_energy']:.3f} +/- {analysis['error']:.3f} kcal/mol", flush=True)

    if job["top_up"]:
        subprocess.run(job["top_up"], check=False)


if __name__ == "__main__":
    main()
