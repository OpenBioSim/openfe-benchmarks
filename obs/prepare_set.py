"""Prepare the RBFE and ABFE inputs for the systems in a benchmark set, running
several systems in parallel. Options not listed here, e.g. --output-dir or
--overwrite, are passed on to prepare_rbfe.py and prepare_abfe.py."""

import argparse
import logging
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openfe_benchmarks.data import BenchmarkIndex

import _sets

logger = logging.getLogger("obs")

SCRIPTS = {kind: Path(__file__).parent / f"prepare_{kind}.py" for kind in _sets.KINDS}


def commands(entries, extra):
    """Group the commands by system, since jobs for the same system share its
    parameter cache and must run in turn."""
    index = BenchmarkIndex()
    by_system = defaultdict(list)
    for block, kind, group, name, options in entries:
        if name not in index.list_system_names_by_group(group):
            raise ValueError(f"No system {group}/{name} (block {block})")
        command = [sys.executable, str(SCRIPTS[kind])]
        command += ["--system-group", group, "--system-name", name]
        command += [f"--{k.replace('_', '-')}" for k, v in options.items() if v]
        command += extra
        if command not in by_system[(group, name)]:
            by_system[(group, name)].append(command)
    return by_system


def run_system(system, system_commands, log_dir):
    log = log_dir / f"{system[0]}_{system[1]}.log"
    start = time.time()
    failed = False
    with open(log, "w") as f:
        for command in system_commands:
            f.write(f"$ {' '.join(command)}\n")
            f.flush()
            result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT)
            failed |= result.returncode != 0
    return system, failed, (time.time() - start) / 60, log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", default="curated", help="Set name or YAML file.")
    parser.add_argument("--blocks", nargs="+", default=None, help="Default: all blocks.")
    parser.add_argument("--jobs", type=int, default=1, help="Systems to prepare at once.")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--dry-run", action="store_true", help="List the commands and exit.")
    args, extra = parser.parse_known_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    by_system = commands(_sets.load_set(args.set, args.blocks), extra)
    logger.info(f"Preparing {len(by_system)} systems from '{args.set}'")
    if args.dry_run:
        for system_commands in by_system.values():
            for command in system_commands:
                print(" ".join(command[1:]))
        return

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_system, s, c, log_dir) for s, c in by_system.items()]
        for i, future in enumerate(as_completed(futures), 1):
            system, system_failed, minutes, log = future.result()
            name = "/".join(system)
            if system_failed:
                failed.append(name)
                logger.error(f"[{i}/{len(futures)}] {name} had failures, see {log}")
            else:
                logger.info(f"[{i}/{len(futures)}] {name} prepared in {minutes:.1f} min")

    if failed:
        logger.error(f"{len(failed)} of {len(by_system)} systems had failures: {', '.join(failed)}")
        raise SystemExit(1)
    logger.info(f"All {len(by_system)} systems prepared")


if __name__ == "__main__":
    main()
