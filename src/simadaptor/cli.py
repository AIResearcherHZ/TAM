from __future__ import annotations

import sys
from typing import Sequence, TypeVar

import tyro

T = TypeVar("T")


def normalize_tyro_argv(argv: Sequence[str]) -> list[str]:
    """Normalize Tyro long-option spellings before parsing.

    Tyro accepts underscore aliases for long flags, but variadic options only
    stop consuming arguments reliably at the canonical hyphenated spellings.
    Normalizing here keeps existing configs working while avoiding greedy nargs
    parse failures such as `--hz_filter 200 500 1000 --robot_key panda`.
    """

    normalized: list[str] = []
    passthrough = False
    for arg in argv:
        if passthrough:
            normalized.append(arg)
            continue
        if arg == "--":
            passthrough = True
            normalized.append(arg)
            continue
        if not arg.startswith("--") or arg.startswith("---"):
            normalized.append(arg)
            continue

        flag, sep, value = arg.partition("=")
        flag = flag.replace("_", "-")
        normalized.append(flag if not sep else f"{flag}={value}")
    return normalized


def parse_tyro_config(config_type: type[T], *, args: Sequence[str] | None = None) -> T:
    raw_args = sys.argv[1:] if args is None else list(args)
    return tyro.cli(config_type, args=normalize_tyro_argv(raw_args))
