#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

"""Validate commit messages against KleidiCV rules.

Supports:
- --range <git-range>  : validate all commits in range
- --file <path>        : validate a commit message file (commit-msg hook)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Iterable

MAX_LINE_LEN = 72
WARN_SUBJECT_LEN = 50
ASCII_MAX = 0x7F


def _run_git_log(commit_range: str) -> str:
    result = subprocess.run(
        ["git", "log", f"--format=%x1e%H%x1f%B", commit_range],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout


def _iter_commits(commit_range: str) -> Iterable[tuple[str, str]]:
    output = _run_git_log(commit_range)
    for record in output.split("\x1e"):
        if not record.strip():
            continue
        if "\x1f" not in record:
            continue
        commit, message = record.split("\x1f", 1)
        yield commit.strip(), message


def _read_message_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


def _remove_comment_lines(message: str) -> list[str]:
    lines = []
    for raw in message.splitlines():
        if raw.lstrip().startswith("#"):
            continue
        lines.append(raw.rstrip("\r"))
    return lines


def _check_message(message: str) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    lines = _remove_comment_lines(message)
    if not lines or all(line.strip() == "" for line in lines):
        errors.append("subject line is empty")
        return errors, warnings

    subject = lines[0].rstrip()
    if subject.strip() == "":
        errors.append("subject line is empty")
        return errors, warnings

    # Subject line ASCII-only check
    if any(ord(ch) > ASCII_MAX for ch in subject):
        errors.append("subject line contains non-ASCII characters")

    # Subject line length checks
    if len(subject) > MAX_LINE_LEN:
        errors.append(
            f"subject line exceeds {MAX_LINE_LEN} chars ({len(subject)})"
        )
    elif len(subject) > WARN_SUBJECT_LEN:
        warnings.append(
            f"subject line exceeds {WARN_SUBJECT_LEN} chars ({len(subject)})"
        )

    # Subject line formatting
    if subject and subject[0].islower():
        warnings.append("subject line should start with a capital letter")

    if subject.endswith("."):
        errors.append("subject line must not end with a period")

    # Blank line between subject and body
    if len(lines) > 1 and lines[1].strip() != "":
        errors.append("missing blank line between subject and body")

    for idx, line in enumerate(lines[1:], start=2):
        # Body line ASCII-only check
        if any(ord(ch) > ASCII_MAX for ch in line):
            errors.append(f"line {idx} contains non-ASCII characters")
        # Body line length checks
        if len(line) > MAX_LINE_LEN:
            errors.append(
                f"line {idx} exceeds {MAX_LINE_LEN} chars ({len(line)})"
            )

    return errors, warnings


def _report_result(
    label: str, errors: list[str], warnings: list[str], quiet: bool = False
) -> int:
    if quiet:
        return 1 if errors else 0
    if warnings:
        for warning in warnings:
            print(f"warning: {label}: {warning}", file=sys.stderr)
    if errors:
        for error in errors:
            print(f"error: {label}: {error}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate commit messages against KleidiCV rules."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--range", help="git commit range to validate")
    group.add_argument("--file", help="commit message file to validate")
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="suppress output"
    )
    args = parser.parse_args()

    exit_code = 0

    if args.range:
        try:
            commits = list(_iter_commits(args.range))
        except subprocess.CalledProcessError as exc:
            if not args.quiet:
                print(exc.stderr.strip(), file=sys.stderr)
            return 2

        if not commits:
            if not args.quiet:
                print("warning: no commits found in range", file=sys.stderr)
            return 0

        for commit, message in commits:
            errors, warnings = _check_message(message)
            exit_code |= _report_result(commit, errors, warnings, args.quiet)
    else:
        message = _read_message_file(args.file)
        errors, warnings = _check_message(message)
        exit_code |= _report_result(args.file, errors, warnings, args.quiet)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
