"""Parameterise the protein, ligands and cofactors of an openfe-benchmarks system."""

import argparse

import _common


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    _common.add_common_args(parser)
    args = parser.parse_args()

    _common.setup_logging()
    out = _common.parameterise_system(args)
    _common.logger.info(f"Parameters written to {out}")


if __name__ == "__main__":
    main()
