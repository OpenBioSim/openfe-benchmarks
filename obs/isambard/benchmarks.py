"""Plan, submit, monitor and collect SOMD2 benchmark runs on Isambard-AI, as set out
in config.yaml. This runs on a login node: it reads files, calls squeue once, and
submits jobs. The simulations and their analysis run inside the jobs (leg.py)."""

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _sets  # noqa: E402

LEGS = ("bound", "free")


def expand(value):
    return os.path.expandvars(os.path.expanduser(str(value)))


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    config["paths"] = {k: Path(expand(v)) for k, v in config["paths"].items()}
    config["file"] = str(Path(path).resolve())
    return config


def somd2_config(name, leg):
    """The SOMD2 config for a leg, with any 'legs' overrides for it applied."""
    config = yaml.safe_load((HERE / "somd2" / f"{name}.yaml").read_text())
    config.update((config.pop("legs", None) or {}).get(leg, {}))
    return config


@dataclass(frozen=True)
class Leg:
    block: str
    kind: str
    group: str
    system: str
    name: str
    leg: str
    somd2: str
    replicate: int
    input_dir: Path

    @property
    def id(self):
        return "/".join(
            [self.group, self.system, self.kind, self.name, self.somd2, self.leg, f"rep{self.replicate}"]
        )

    @property
    def job_name(self):
        return "obs." + self.id.replace("/", ".")


def find_legs(config):
    """Every leg and replicate selected by the config, in block order."""
    root = config["paths"]["input"]
    only = set(config.get("only") or [])
    legs = []
    for block, kind, group, system, options in _sets.load_set(
        config["set"], list(config["blocks"])
    ):
        somd2 = config["blocks"][block]
        system_dir = root / group / system / kind
        if not system_dir.is_dir():
            print(f"Warning: no inputs for {group}/{system} ({kind}) in {root}", file=sys.stderr)
            continue
        for d in sorted(p for p in system_dir.iterdir() if p.is_dir()):
            if only and f"{group}/{system}" not in only and f"{group}/{system}/{d.name}" not in only:
                continue
            if kind == "rbfe":
                info = json.loads((d / "edge.json").read_text())
                from_file = info.get("mapping_method", {}).get("mapper") == "file"
                if from_file != bool(options.get("mappings")):
                    continue
            for leg in LEGS:
                if not (d / f"{leg}_lambda_0.bss").exists():
                    print(f"Warning: {d}/{leg}_lambda_0.bss is missing", file=sys.stderr)
                    continue
                for rep in range(1, config["replicates"] + 1):
                    legs.append(Leg(block, kind, group, system, d.name, leg, somd2, rep, d))
    return legs


def queue():
    """Our benchmark jobs in the queue, as {job name: state}, excluding the calling
    job when run from inside one."""
    own = os.environ.get("SLURM_JOB_ID")
    result = subprocess.run(
        ["squeue", "--me", "--noheader", "--Format=JobID:20,Name:255,State:20"],
        capture_output=True,
        text=True,
        check=True,
    )
    jobs = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[1].startswith("obs.") and fields[0] != own:
            jobs[fields[1]] = fields[2]
    return jobs


def run_dir(config, leg):
    return config["paths"]["scratch"] / leg.id


def output_dir(config, leg):
    return config["paths"]["output"] / leg.id


def attempts(config, leg):
    return sorted(run_dir(config, leg).glob("attempt.*"))


def state(config, leg, queued):
    if (output_dir(config, leg) / "analysis.json").exists():
        return "done"
    if leg.job_name in queued:
        return queued[leg.job_name].lower()
    n = len(attempts(config, leg))
    if n == 0:
        return "new"
    return "failed" if n >= config["slurm"]["max_attempts"] else "unfinished"


def gpu_hours(config, leg):
    somd2 = somd2_config(leg.somd2, leg.leg)
    match = re.fullmatch(r"\s*([0-9.eE+-]+)\s*(fs|ps|ns|us)\s*", str(somd2["runtime"]))
    if match is None:
        raise ValueError(f"Can't read the runtime '{somd2['runtime']}' of '{leg.somd2}'")
    scale = {"fs": 1e-6, "ps": 1e-3, "ns": 1.0, "us": 1e3}[match.group(2)]
    runtime = float(match.group(1)) * scale
    num_lambda = somd2.get("num_lambda", 11)
    return runtime * num_lambda / config["throughput"][leg.leg] * 24 * config["slurm"]["gpus_per_job"]


def submit_leg(config, leg, dry_run):
    """Claim the leg's next attempt, so that concurrent callers can't both submit
    it, then write its job files and submit it."""
    directory = run_dir(config, leg)
    directory.mkdir(parents=True, exist_ok=True)
    attempt = directory / f"attempt.{len(attempts(config, leg)) + 1}"
    if dry_run:
        print(f"Would submit {leg.id}")
        return True
    try:
        os.close(os.open(attempt, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        return False

    somd2 = somd2_config(leg.somd2, leg.leg)
    fallback = directory / "fallback_timestep"
    if fallback.exists():
        somd2["timestep"] = fallback.read_text().strip()
    (directory / "somd2.yaml").write_text(yaml.safe_dump(somd2))
    job = {
        "id": leg.id,
        "system": str(leg.input_dir / f"{leg.leg}_lambda_0.bss"),
        "perturbed_system": (
            str(leg.input_dir / f"{leg.leg}_lambda_1.bss") if leg.kind == "rbfe" else None
        ),
        "output": str(output_dir(config, leg)),
        "copy_trajectories": bool(config.get("copy_trajectories", False)),
        "fallback_timestep": config.get("fallback_timestep"),
        "top_up": (
            [sys.executable, str(HERE / "benchmarks.py"), "--config", config["file"], "submit"]
            if config.get("top_up", True)
            else None
        ),
    }
    (directory / "job.json").write_text(json.dumps(job, indent=2))
    (directory / "setup.sh").write_text("\n".join(config["setup"]) + "\n")

    slurm = config["slurm"]
    command = [
        "sbatch",
        "--parsable",
        f"--job-name={leg.job_name}",
        "--nodes=1",
        f"--gpus={slurm['gpus_per_job']}",
        f"--time={slurm['time']}",
        f"--output={directory}/slurm-%j.out",
        *slurm.get("extra", []),
        str(HERE / "leg.sbatch"),
        str(directory),
        str(HERE / "leg.py"),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        attempt.unlink()
        print(f"Failed to submit {leg.id}: {result.stderr.strip()}", file=sys.stderr)
        return False
    attempt.write_text(result.stdout.strip() + "\n")
    print(f"Submitted {leg.id} as job {result.stdout.strip()}")
    return True


def cmd_plan(config, args):
    queued = queue()
    legs = find_legs(config)
    counts = defaultdict(Counter)
    remaining = defaultdict(float)
    for leg in legs:
        s = state(config, leg, queued)
        counts[leg.block][s] += 1
        if s != "done":
            remaining[leg.block] += gpu_hours(config, leg)
    for block in config["blocks"]:
        summary = ", ".join(f"{n} {s}" for s, n in sorted(counts[block].items()))
        print(f"{block} ({config['blocks'][block]}): {summary or 'no legs'}; "
              f"~{remaining[block]:.0f} GPU-h to go")
    print(f"Total ~{sum(remaining.values()):.0f} GPU-h to go for {len(legs)} legs")
    problems = [leg for leg in legs if state(config, leg, queued) in ("unfinished", "failed")]
    for leg in problems:
        logs = sorted(run_dir(config, leg).glob("slurm-*.out"))
        print(f"  {state(config, leg, queued)}: {leg.id} ({logs[-1] if logs else 'no log'})")


def cmd_submit(config, args):
    queued = queue()
    slots = config["slurm"]["max_jobs"] - len(queued)
    submitted = 0
    for leg in find_legs(config):
        if slots <= 0:
            break
        if state(config, leg, queued) in ("new", "unfinished") and submit_leg(config, leg, args.dry_run):
            slots -= 1
            submitted += 1
    print(f"{'Would submit' if args.dry_run else 'Submitted'} {submitted} jobs")


def cmd_collect(config, args):
    results = defaultdict(lambda: defaultdict(dict))
    for leg in find_legs(config):
        path = output_dir(config, leg) / "analysis.json"
        if path.exists():
            key = (leg.block, leg.kind, leg.group, leg.system, leg.name, leg.somd2)
            results[key][leg.replicate][leg.leg] = json.loads(path.read_text())

    rows = []
    for (block, kind, group, system, name, somd2), reps in sorted(results.items()):
        values, errors, overlaps = [], [], []
        for rep in sorted(reps):
            legs = reps[rep]
            if not all(leg in legs for leg in LEGS):
                continue
            bound, free = legs["bound"], legs["free"]
            if kind == "rbfe":
                value = bound["free_energy"] - free["free_energy"]
            else:
                value = free["free_energy"] - bound["free_energy"] - (bound.get("standard_state_correction") or 0.0)
            values.append(value)
            errors.append((bound["error"] ** 2 + free["error"] ** 2) ** 0.5)
            overlaps.append(min(bound["min_overlap"], free["min_overlap"]))
        if not values:
            continue
        rows.append({
            "block": block, "kind": kind, "system": f"{group}/{system}", "name": name, "config": somd2,
            "replicates": len(values),
            "mean": round(statistics.mean(values), 3),
            "sem": round(statistics.stdev(values) / len(values) ** 0.5, 3) if len(values) > 1 else "",
            "mbar_error": round(statistics.mean(errors), 3),
            "min_overlap": round(min(overlaps), 4),
            "values": " ".join(f"{v:.3f}" for v in values),
        })

    out = Path(args.output or config["paths"]["output"] / "results.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["block"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} results to {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan", help="Show progress, problems and the GPU-hours still to run.")
    p = sub.add_parser("submit", help="Submit legs, up to the slurm 'max_jobs' limit.")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("collect", help="Combine finished legs into free energies (CSV).")
    p.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    {"plan": cmd_plan, "submit": cmd_submit, "collect": cmd_collect}[args.command](config, args)


if __name__ == "__main__":
    main()
