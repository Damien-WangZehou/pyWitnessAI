"""Support ``python -m pyWitnessAI`` with the same installed command entry."""

from .API.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
