"""Generate cddlib's "_f" (float/myfloat) variant of a .c/.h file.

Portable replacement for cddlib's own lib-src/Makefile.gmp.am SED_GMP
rule + its trailing awk header-comment step (sed/awk aren't guaranteed
present on Windows, but this repo already depends on Python everywhere).
Must replicate sed's per-line, in-order, first-match-unless-/g semantics
exactly -- in particular the dd_ -> ddf_ (global) immediately followed by
cddf_ -> cdd_ (global) pair only makes sense applied in that order: it
renames bona fide dd_-prefixed identifiers to ddf_, while undoing the
accidental dd_ -> ddf_ hit inside cdd_-prefixed identifiers (cdd_ contains
the substring dd_) so those round-trip back to cdd_ unchanged.
"""
import argparse
import sys
from pathlib import Path

# (pattern, replacement, is_global) in the exact order cddlib's SED_GMP applies them.
RULES: list[tuple[str, str, bool]] = [
    ("dd_", "ddf_", True),
    ("cddf_", "cdd_", True),
    ("mytype", "myfloat", True),
    ('#include "cdd.h"', '#include "cdd_f.h"', False),
    ('#include "cddtypes.h"', '#include "cddtypes_f.h"', False),
    ('#include "cddmp.h"', '#include "cddmp_f.h"', False),
    ("__CDD_H", "__CDD_HF", False),
    ("__CDD_HFF", "__CDD_HF", False),
    ("__CDDMP_H", "_CDDMP_HF", False),
    ("__CDDTYPES_H", "_CDDTYPES_HF", False),
    ("GMPRATIONAL", "ddf_GMPRATIONAL", True),
    ("ARITHMETIC", "ddf_ARITHMETIC", True),
    ("CDOUBLE", "ddf_CDOUBLE", True),
]


def transform_line(line: str) -> str:
    for pattern, replacement, is_global in RULES:
        if is_global:
            line = line.replace(pattern, replacement)
        else:
            idx = line.find(pattern)
            if idx != -1:
                line = line[:idx] + replacement + line[idx + len(pattern):]
    return line


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    args = parser.parse_args()

    # newline="": preserve the source's own line endings verbatim (splitlines(keepends=True)
    # keeps each line's original terminator) instead of letting Python's text-mode write
    # translate \n to the platform default (CRLF on Windows). The prepended header line must
    # match that same convention too, or the file ends up with mixed CRLF/LF terminators.
    text = args.src.read_text(newline="")
    eol = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines(keepends=True)
    transformed = "".join(transform_line(line) for line in lines)

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    args.dst.write_text(
        f"/* generated automatically from {args.src.name} */{eol}{transformed}", newline=""
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
