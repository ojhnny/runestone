"""Command line entry point: ``so-ell <stage>`` or ``so-ell all``."""

from __future__ import annotations

import argparse

from so_ell import config, ingest, pipeline


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="so-ell", description="SO-ELL ebook engagement pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("manifest", help="Rebuild data/raw/MANIFEST.csv after adding or replacing raw exports")
    run_all = sub.add_parser("all", help="Run every stage in order")
    run_all.add_argument("--from", dest="start", default="ingest", choices=list(pipeline.STAGES))
    for name in pipeline.STAGES:
        stage = sub.add_parser(name, help=f"Run only the {name} stage")
        if name == "ingest":
            stage.add_argument("--skip-checksums", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "manifest":
        print(ingest.build_manifest(config.RAW_DIR).to_string(index=False))
    elif args.command == "all":
        pipeline.run_all(args.start)
    elif args.command == "ingest":
        pipeline.stage_ingest(verify_checksums=not args.skip_checksums)
    else:
        pipeline.STAGES[args.command]()


if __name__ == "__main__":
    main()
