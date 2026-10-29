"""Lightweight command entry point; feature modules load only when requested."""

from __future__ import annotations

import argparse
from importlib import import_module
import json
import sys

from .. import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyWitnessAI",
        description="Description processing and lineup workflows.",
    )
    parser.add_argument("--version", action="version", version=f"pyWitnessAI {__version__}")
    commands = parser.add_subparsers(dest="command")
    parse = commands.add_parser("parse", help="Preprocess a description and print its JSON audit.")
    parse.add_argument("description", help="Witness description, quoted as a single argument.")
    parse.add_argument("--parser", choices=("raw", "lenient", "strict", "schema"), default="raw")
    parse.add_argument("--preserve-case", action="store_true", help="Preserve case in model text.")
    serve = commands.add_parser(
        "serve",
        help="Start the lineup and session HTTP API.",
        description=(
            "HOST, PORT and LOG_LEVEL configure Uvicorn. "
            'Install pyWitnessAI[api,generation] for lineup generation.'
        ),
    )
    serve.add_argument("--image-root", default=".", help="Allowed input image directory for the lineup API.")
    serve.add_argument("--output-dir", default="./lineup_service", help="Lineup API state and presentation directory.")
    serve.add_argument("--allow-origin", action="append", default=[], help="Allowed lineup API browser origin; repeat for multiple origins.")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(arguments)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "serve":
        import os
        app = import_module("pyWitnessAI.API.http").create_app(
            output_dir=args.output_dir, image_root=args.image_root, allowed_origins=args.allow_origin,
        )
        import_module("uvicorn").run(
            app, host=os.getenv("HOST", "127.0.0.1"),
            port=int(os.getenv("PORT", "8000")), log_level=os.getenv("LOG_LEVEL", "info"),
        )
        return 0

    from ..Description import create_description_parser

    parsed = create_description_parser(args.parser, lowercase=not args.preserve_case).parse(args.description)
    print(json.dumps(parsed.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
