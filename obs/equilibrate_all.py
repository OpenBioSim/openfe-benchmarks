"""Equilibrate every prepared input under an output directory, running one job at a
time on each GPU. Bound legs are run first, since they take longest."""

import argparse
import fnmatch
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

logger = logging.getLogger("obs")

EQUILIBRATE = Path(__file__).parent / "equilibrate.py"
END_STATES = {"rbfe": [0, 1], "abfe": [0]}


def find_jobs(root, kinds, systems, overwrite):
    jobs = []
    for kind in kinds:
        for path in sorted(Path(root).glob(f"*/*/{kind}/*/*.bss")):
            if "_lambda_" in path.stem:
                continue
            group, name = path.parts[-5], path.parts[-4]
            if systems and not any(fnmatch.fnmatch(f"{group}/{name}", s) for s in systems):
                continue
            for state in END_STATES[kind]:
                output = path.with_name(f"{path.stem}_lambda_{state}.bss")
                if overwrite or not output.exists():
                    jobs.append((path, state))
    return sorted(jobs, key=lambda job: job[0].stem != "bound")


def worker(gpu, jobs, args, counts, lock):
    # Number GPUs as nvidia-smi does, rather than fastest first.
    env = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", **os.environ}
    env.update({"CUDA_VISIBLE_DEVICES": gpu, "OPENCL_VISIBLE_DEVICES": gpu})
    while True:
        try:
            path, state = jobs.get_nowait()
        except queue.Empty:
            return
        log = path.with_name(f"{path.stem}_lambda_{state}.log")
        command = [sys.executable, str(EQUILIBRATE), str(path), "--end-states", str(state)]
        command += ["--protocol", args.protocol, "--overwrite"]
        if args.platform:
            command += ["--platform", args.platform]

        start = time.time()
        with open(log, "w") as f:
            result = subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT)
        minutes = (time.time() - start) / 60

        with lock:
            counts["done"] += 1
            status = f"[{counts['done']}/{counts['total']}] GPU {gpu}: {path} lambda = {state}"
            if result.returncode == 0:
                logger.info(f"{status} finished in {minutes:.1f} min")
            else:
                counts["failed"].append(f"{path} lambda = {state}")
                logger.error(f"{status} failed after {minutes:.1f} min, see {log}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="obs_output", help="Root output directory.")
    parser.add_argument("--gpus", nargs="+", default=["0"], help="GPU indices, e.g. 0 1 2")
    parser.add_argument("--kinds", nargs="+", default=["rbfe", "abfe"], choices=["rbfe", "abfe"])
    parser.add_argument(
        "--systems", nargs="+", default=None, help="Systems to include, e.g. jacs_set/tyk2 'merck/*'"
    )
    parser.add_argument(
        "--protocol", default=str(Path(__file__).parent / "config" / "equilibration.yaml")
    )
    parser.add_argument("--platform", default=None, help="e.g. cuda, opencl")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="List the jobs and exit.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    found = find_jobs(args.output_dir, args.kinds, args.systems, args.overwrite)
    logger.info(f"{len(found)} end states to equilibrate on GPUs {', '.join(args.gpus)}")
    if args.dry_run:
        for path, state in found:
            print(f"{path} lambda = {state}")
        return

    jobs = queue.Queue()
    for job in found:
        jobs.put(job)
    counts = {"done": 0, "total": len(found), "failed": []}
    lock = threading.Lock()
    threads = [
        threading.Thread(target=worker, args=(gpu, jobs, args, counts, lock), daemon=True)
        for gpu in args.gpus
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if counts["failed"]:
        logger.error(f"{len(counts['failed'])} of {len(found)} failed:")
        for job in counts["failed"]:
            logger.error(f"  {job}")
        raise SystemExit(1)
    logger.info(f"All {len(found)} end states equilibrated")


if __name__ == "__main__":
    main()
