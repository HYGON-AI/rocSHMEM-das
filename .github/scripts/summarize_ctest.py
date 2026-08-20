#!/usr/bin/env python3

import argparse
import re
import sys
from pathlib import Path


def markdown_cell(value):
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def backend_from_name(path):
    match = re.search(r"_(ipc|gda|ro)_[^/\\]+\.log$", path.name, re.IGNORECASE)
    return match.group(1).upper() if match else "unknown"


def parse_results(report_dir):
    cases = []
    errors = []
    ansi_escape = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    result_line = re.compile(
        r"^\s*\d+/\d+\s+Test\s+#\d+:\s+(.+?)\s+(?:\.{2,}\s*)?"
        r"(Passed|\*{3}Failed|\*{3}Timeout|\*{3}Not Run|Failed|Timeout|Not Run)"
        r"(?:\s+|$)"
    )

    for log_path in sorted(report_dir.glob("test_*.log")):
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            errors.append(f"{log_path.name}: {exc}")
            continue

        backend = backend_from_name(log_path)
        for line in lines:
            match = result_line.match(ansi_escape.sub("", line))
            if not match:
                continue
            result = match.group(2).lstrip("*")
            status = "passed" if result == "Passed" else "failed"
            if result == "Not Run":
                status = "skipped"
            cases.append(
                {
                    "backend": backend,
                    "name": match.group(1).rstrip(". "),
                    "status": status,
                }
            )
    return cases, errors


def render_summary(cases, errors, suite):
    total = len(cases)
    failed = [case for case in cases if case["status"] == "failed"]
    skipped = [case for case in cases if case["status"] == "skipped"]
    passed = total - len(failed) - len(skipped)

    lines = [
        "# rocSHMEM Test Results",
        "",
        f"Suite: `{suite}`",
        "",
        "| Executed | Passed | Failed | Skipped |",
        "| ---: | ---: | ---: | ---: |",
        f"| {total} | {passed} | {len(failed)} | {len(skipped)} |",
        "",
    ]

    if failed:
        lines.extend(
            [
                "## Failed Tests",
                "",
                "| Backend | Test case |",
                "| --- | --- |",
            ]
        )
        for case in failed:
            lines.append(
                f"| {markdown_cell(case['backend'])} | {markdown_cell(case['name'])} |"
            )
        lines.append("")
    elif total:
        lines.extend(["All executed tests passed.", ""])
    else:
        lines.extend(["No CTest result lines were found in the report directory.", ""])

    if errors:
        lines.extend(["## Report Errors", ""])
        lines.extend(f"- {markdown_cell(error)}" for error in errors)
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Create a GitHub summary from CTest logs")
    parser.add_argument("report_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--suite", default="unknown")
    args = parser.parse_args()

    cases, errors = parse_results(args.report_dir)
    summary = render_summary(cases, errors, args.suite)
    args.output.write_text(summary, encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
