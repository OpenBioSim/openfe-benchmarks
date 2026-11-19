"""Scan prepared RBFE edges for vanishing/appearing ghost pairs that are 1-4 (or
otherwise scaled) neighbours, which kept a hard Coulomb interaction at intermediate
lambda before the Sire fix for to/from ghost exceptions (OpenBioSim/sire#540).

    python obs/scan_ghost_pairs.py <input root> [--legs free bound] [--jobs 8]
        [--output ghost_pairs.csv] [--top 30]

The input root is laid out as <group>/<system>/rbfe/<edge>/<leg>_lambda_<n>.bss,
e.g. obs_output or the Isambard input directory. For each pair the charge product is
taken at lambda = 0.5, where it peaks for a linear charge morph, times its 1-4 scale.
The energy is that charge product over the closest approach of the pair in the
equilibrated end states. One CSV row is written per pair, and the edges are printed
ranked by their largest energy.
"""

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

COULOMB = 332.0637  # kcal/mol Å / e^2


def scan_edge(edge, legs):
    import sire as sr

    def load(path):
        return sr.stream.load(str(path)).molecules("property is_perturbable")[0]

    mol = load(edge / f"{legs[0]}_lambda_0.bss")
    el0, el1 = mol.property("element0"), mol.property("element1")
    q0, q1 = mol.property("charge0"), mol.property("charge1")
    s0, s1 = mol.property("intrascale0"), mol.property("intrascale1")
    n = mol.num_atoms()
    vanish = [i for i in range(n) if el1[i].symbol() == "Xx"]
    appear = [j for j in range(n) if el0[j].symbol() == "Xx"]

    pairs = []
    for i in vanish:
        for j in appear:
            a, b = sr.mol.AtomIdx(i), sr.mol.AtomIdx(j)
            scale = max(s0.get(a, b).coulomb(), s1.get(a, b).coulomb())
            if 0 < scale < 1:
                qq = 0.25 * q0[i].value() * q1[j].value() * scale
                pairs.append([i, j, el0[i].symbol(), el1[j].symbol(), scale, qq, np.inf])
    if not pairs:
        return edge, []

    for leg in legs:
        for end in (0, 1):
            path = edge / f"{leg}_lambda_{end}.bss"
            if not path.exists():
                continue
            x = np.array(
                [[c.x().value(), c.y().value(), c.z().value()] for c in load(path).property(f"coordinates{end}").to_vector()]
            )
            for p in pairs:
                p[6] = min(p[6], float(np.linalg.norm(x[p[0]] - x[p[1]])))

    return edge, [p + [COULOMB * p[5] / p[6]] for p in pairs]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("--legs", nargs="+", default=["free", "bound"])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("ghost_pairs.csv"))
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    edges = sorted(p.parent for p in args.root.glob(f"*/*/rbfe/*/{args.legs[0]}_lambda_0.bss"))
    print(f"Scanning {len(edges)} edges", file=sys.stderr)

    results = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(scan_edge, edge, args.legs) for edge in edges]
        for k, future in enumerate(futures, 1):
            try:
                edge, pairs = future.result()
                results[edge] = pairs
            except Exception as e:
                print(f"Failed {edges[k - 1]}: {e}", file=sys.stderr)
            if k % 50 == 0:
                print(f"{k}/{len(edges)}", file=sys.stderr)

    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["edge", "vanishing", "appearing", "element_a", "element_b", "scale", "qq_half", "min_distance", "energy_half"])
        for edge, pairs in results.items():
            name = str(edge.relative_to(args.root))
            for i, j, ea, eb, scale, qq, r, e in pairs:
                w.writerow([name, i, j, ea, eb, round(scale, 4), round(qq, 5), round(r, 3), round(e, 3)])

    affected = {e: p for e, p in results.items() if p}
    print(f"{len(affected)} of {len(results)} edges have vanishing/appearing 1-4 pairs; pairs written to {args.output}")
    ranked = sorted(affected.items(), key=lambda kv: -max(abs(p[7]) for p in kv[1]))
    print(f"\n{'edge':60s} {'pairs':>5s} {'max |qq|':>9s} {'min r (Å)':>9s} {'max |E| (kcal/mol)':>18s}")
    for edge, pairs in ranked[: args.top]:
        worst = max(pairs, key=lambda p: abs(p[7]))
        print(f"{str(edge.relative_to(args.root)):60s} {len(pairs):5d} {max(abs(p[5]) for p in pairs):9.4f} "
              f"{min(p[6] for p in pairs):9.2f} {abs(worst[7]):18.2f}")


if __name__ == "__main__":
    main()
