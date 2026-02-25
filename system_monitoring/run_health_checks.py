#!/usr/bin/env python3
"""Run configurable microkernel health checks from a YAML file."""

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


class ConfigError(Exception):
    """Raised when YAML config is missing required fields or is malformed."""


def parse_csv(arg: str) -> List[str]:
    if not arg:
        return []
    return [item.strip() for item in arg.split(",") if item.strip()]


def load_yaml(config_path: Path) -> Dict:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise ConfigError(
            "PyYAML is required. Install with `pip install pyyaml`."
        ) from exc

    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ConfigError("Top-level YAML must be a mapping/object.")

    checks = data.get("checks", [])
    if not isinstance(checks, list):
        raise ConfigError("`checks` must be a list.")

    for i, check in enumerate(checks):
        if not isinstance(check, dict):
            raise ConfigError(f"checks[{i}] must be a mapping/object")
        if "id" not in check:
            raise ConfigError(f"checks[{i}] is missing required field `id`")
        if "command" not in check:
            raise ConfigError(f"checks[{i}] is missing required field `command`")

    return data


def apply_template(value: str, template_vars: Dict[str, str]) -> str:
    try:
        return value.format(**template_vars)
    except KeyError as exc:
        raise ConfigError(f"Unknown template variable in command/cwd: {exc}") from exc


def render_command(command, template_vars: Dict[str, str]):
    if isinstance(command, str):
        return apply_template(command, template_vars), True
    if isinstance(command, list):
        rendered = [apply_template(str(part), template_vars) for part in command]
        return rendered, False
    raise ConfigError("`command` must be a string or list.")


def should_include(
    check: Dict,
    include_groups: Iterable[str],
    include_checks: Iterable[str],
    exclude_checks: Iterable[str],
    include_disabled: bool,
) -> bool:
    check_id = str(check.get("id", "")).strip()
    check_group = str(check.get("group", "")).strip()
    enabled = bool(check.get("enabled", True))

    if include_checks and check_id not in include_checks:
        return False
    if include_groups and check_group not in include_groups:
        return False
    if check_id in exclude_checks:
        return False
    if not include_disabled and not enabled:
        return False
    return True


def run_check(
    check: Dict,
    template_vars: Dict[str, str],
    default_env: Dict[str, str],
    dry_run: bool,
) -> Tuple[bool, float, int]:
    check_id = check["id"]
    command, use_shell = render_command(check["command"], template_vars)

    cwd = check.get("cwd")
    run_cwd = None
    if cwd:
        run_cwd = apply_template(str(cwd), template_vars)

    env = os.environ.copy()
    env.update(default_env)
    env.update({str(k): str(v) for k, v in check.get("env", {}).items()})

    timeout = check.get("timeout")

    printable_cmd = command if isinstance(command, str) else " ".join(shlex.quote(c) for c in command)

    print(f"\n=== Running {check_id} ===")
    print(f"Command: {printable_cmd}")
    if run_cwd:
        print(f"Working dir: {run_cwd}")
    if timeout:
        print(f"Timeout: {timeout}s")

    if dry_run:
        print("Dry-run enabled; command not executed.")
        return True, 0.0, 0

    start = time.time()
    try:
        completed = subprocess.run(
            command,
            shell=use_shell,
            cwd=run_cwd,
            env=env,
            timeout=timeout,
            check=False,
        )
        elapsed = time.time() - start
        ok = completed.returncode == 0
        return ok, elapsed, int(completed.returncode)
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"Timed out after {elapsed:.1f}s")
        return False, elapsed, 124


def run_build_step(config: Dict, template_vars: Dict[str, str], dry_run: bool) -> bool:
    build = config.get("build", {})
    if not build:
        return True

    enabled = bool(build.get("enabled", False))
    if not enabled:
        return True

    configure_cmd = build.get(
        "configure_command",
        [
            "cmake",
            "-S",
            "{source_dir}",
            "-B",
            "{build_dir}",
            "-DHEALTH_CHECKS_ENABLE_MPI=ON",
            "-DHEALTH_CHECKS_ENABLE_OPENMP=ON",
        ],
    )
    build_cmd = build.get("build_command", ["cmake", "--build", "{build_dir}", "-j"])

    steps = [
        ("configure", configure_cmd),
        ("build", build_cmd),
    ]

    for step_name, cmd in steps:
        rendered, use_shell = render_command(cmd, template_vars)
        printable_cmd = rendered if isinstance(rendered, str) else " ".join(shlex.quote(c) for c in rendered)
        print(f"\n=== Build step: {step_name} ===")
        print(f"Command: {printable_cmd}")

        if dry_run:
            print("Dry-run enabled; command not executed.")
            continue

        result = subprocess.run(rendered, shell=use_shell, check=False)
        if result.returncode != 0:
            print(f"Build step failed ({step_name}), rc={result.returncode}")
            return False

    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run microkernel health checks from YAML configuration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().with_name("health_checks.yaml")),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--groups",
        default="",
        help="Comma-separated group names to run.",
    )
    parser.add_argument(
        "--checks",
        default="",
        help="Comma-separated check IDs to run.",
    )
    parser.add_argument(
        "--exclude",
        default="",
        help="Comma-separated check IDs to exclude.",
    )
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Include checks that are disabled in YAML.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List selected checks and exit.",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Execute `build` section before running checks.",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Execute only the build section and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after first failed check.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()

    try:
        config = load_yaml(config_path)
    except ConfigError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    repo_root = Path(__file__).resolve().parent.parent
    source_dir = repo_root / "utils" / "check_healthy_tests"
    build_dir = repo_root / "build" / "health_checks"

    build_cfg = config.get("build", {})
    if isinstance(build_cfg, dict):
        raw_source_dir = str(build_cfg.get("source_dir", str(source_dir)))
        raw_build_dir = str(build_cfg.get("build_dir", str(build_dir)))
        source_dir = Path(
            apply_template(raw_source_dir, {"repo_root": str(repo_root)})
        ).resolve()
        build_dir = Path(
            apply_template(
                raw_build_dir,
                {"repo_root": str(repo_root), "source_dir": str(source_dir)},
            )
        ).resolve()

    template_vars = {
        "repo_root": str(repo_root),
        "source_dir": str(source_dir),
        "build_dir": str(build_dir),
    }

    default_env = {str(k): str(v) for k, v in config.get("default_env", {}).items()}

    include_groups = set(parse_csv(args.groups))
    include_checks = set(parse_csv(args.checks))
    exclude_checks = set(parse_csv(args.exclude))

    checks = config.get("checks", [])
    selected = [
        c
        for c in checks
        if should_include(
            c,
            include_groups=include_groups,
            include_checks=include_checks,
            exclude_checks=exclude_checks,
            include_disabled=args.include_disabled,
        )
    ]

    if args.list:
        for c in selected:
            status = "enabled" if c.get("enabled", True) else "disabled"
            group = c.get("group", "ungrouped")
            print(f"- {c['id']} [{group}] ({status})")
        return 0

    if args.build or args.build_only:
        ok = run_build_step(config, template_vars, dry_run=args.dry_run)
        if not ok:
            return 1
        if args.build_only:
            return 0

    if not selected:
        print("No checks selected.")
        return 0

    results: List[Tuple[str, bool, float, int]] = []

    for check in selected:
        ok, elapsed, rc = run_check(
            check,
            template_vars=template_vars,
            default_env=default_env,
            dry_run=args.dry_run,
        )
        check_id = str(check["id"])
        results.append((check_id, ok, elapsed, rc))

        status = "PASS" if ok else "FAIL"
        print(f"Result {check_id}: {status} (rc={rc}, {elapsed:.2f}s)")

        if args.fail_fast and not ok:
            break

    failed = [r for r in results if not r[1]]

    print("\n=== Summary ===")
    for check_id, ok, elapsed, rc in results:
        status = "PASS" if ok else "FAIL"
        print(f"{status:4} {check_id:32} rc={rc:3d} elapsed={elapsed:.2f}s")

    if failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
