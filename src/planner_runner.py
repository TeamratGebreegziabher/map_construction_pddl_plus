from __future__ import annotations

import csv
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any


def ensure_directory(path: str | Path) -> Path:
    """Create directory if it does not exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def check_file_exists(path: str | Path, description: str) -> Path:
    """Check that a required file exists."""
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"{description} not found: {file_path}")

    return file_path


def check_java_available(java_path: str = "java") -> str:
    """
    Check whether Java is available.

    Returns the Java version output.
    """
    try:
        completed = subprocess.run(
            [java_path, "-version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Java executable not found: {java_path}\n"
            "Install Java JDK 17 or newer, or update planner.java_path in config.yaml."
        ) from exc

    version_output = completed.stderr.strip() or completed.stdout.strip()

    if completed.returncode != 0:
        raise RuntimeError(
            "Java exists but did not run successfully.\n"
            f"Command: {java_path} -version\n"
            f"Output:\n{version_output}"
        )

    return version_output


def get_problem_files(config: dict[str, Any]) -> list[Path]:
    """Return generated PDDL problem files in small/medium/large order."""
    problem_dir = Path(config["planning"]["problem_dir"])

    problem_files = [
        problem_dir / "problem_small.pddl",
        problem_dir / "problem_medium.pddl",
        problem_dir / "problem_large.pddl",
    ]

    for problem_file in problem_files:
        check_file_exists(problem_file, "PDDL problem file")

    return problem_files


def build_enhsp_command(
    config: dict[str, Any],
    domain_file: Path,
    problem_file: Path,
) -> list[str]:
    """
    Build ENHSP command.

    Most ENHSP JAR distributions use:
        java -jar enhsp.jar -o domain.pddl -f problem.pddl

    If your local ENHSP version requires extra flags, add them to:
        planner.extra_args
    in config.yaml.
    """
    planner_config = config["planner"]

    java_path = planner_config.get("java_path", "java")
    jar_path = planner_config["jar_path"]
    extra_args = planner_config.get("extra_args", [])

    command = [
        java_path,
        "-jar",
        jar_path,
        "-o",
        str(domain_file),
        "-f",
        str(problem_file),
    ]

    command.extend(extra_args)

    return command


def run_single_problem(
    config: dict[str, Any],
    domain_file: Path,
    problem_file: Path,
) -> dict[str, Any]:
    """
    Run ENHSP on one problem file and return execution result.
    """
    instance_name = problem_file.stem.replace("problem_", "")

    logs_dir = ensure_directory(config["outputs"]["logs_dir"])
    plans_dir = ensure_directory(config["outputs"]["plans_dir"])

    log_path = logs_dir / f"{instance_name}_planner.log"
    plan_path = plans_dir / f"{instance_name}_plan.txt"

    timeout_seconds = int(config["planner"].get("timeout_seconds", 120))

    command = build_enhsp_command(config, domain_file, problem_file)

    print(f"\nRunning planner for instance: {instance_name}")
    print("Command:", " ".join(command))

    start_time = time.perf_counter()

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )

        runtime_seconds = time.perf_counter() - start_time
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        full_output = stdout + "\n" + stderr

        status = "success" if completed.returncode == 0 else "failed"

    except subprocess.TimeoutExpired as exc:
        runtime_seconds = time.perf_counter() - start_time
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""

        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")

        full_output = stdout + "\n" + stderr
        full_output += f"\n\nPlanner timed out after {timeout_seconds} seconds.\n"

        status = "timeout"
        completed = None

    log_path.write_text(full_output, encoding="utf-8")

    plan_lines = extract_plan_lines(full_output)
    plan_found = len(plan_lines) > 0

    if plan_found:
        plan_text = "\n".join(plan_lines) + "\n"
    else:
        plan_text = (
            f"No plan lines were detected for instance '{instance_name}'.\n"
            f"Check the full log file: {log_path}\n"
        )

    plan_path.write_text(plan_text, encoding="utf-8")

    result = {
        "instance_name": instance_name,
        "domain_file": str(domain_file),
        "problem_file": str(problem_file),
        "status": status,
        "return_code": completed.returncode if completed is not None else None,
        "runtime_seconds": round(runtime_seconds, 4),
        "plan_found": plan_found,
        "plan_length": len(plan_lines),
        "log_file": str(log_path),
        "plan_file": str(plan_path),
        "command": command,
    }

    print(
        f"Finished {instance_name}: "
        f"status={status}, "
        f"plan_found={plan_found}, "
        f"plan_length={len(plan_lines)}, "
        f"runtime={runtime_seconds:.3f}s"
    )

    return result


def extract_plan_lines(planner_output: str) -> list[str]:
    """
    Extract likely plan/action lines from planner output.

    This is intentionally tolerant because different ENHSP versions print
    plans slightly differently.

    Supported examples:
        0.000: (start-move car1 loc_2 loc_1)
        170.264: (arrive car1 loc_1)
        (start-move car1 loc_2 loc_1)
    """
    plan_lines: list[str] = []

    timed_action_pattern = re.compile(
        r"^\s*\d+(\.\d+)?\s*:\s*\([a-zA-Z0-9_\-]+.*\)\s*$"
    )

    plain_action_pattern = re.compile(
        r"^\s*\([a-zA-Z0-9_\-]+(\s+[a-zA-Z0-9_\-]+)*\)\s*$"
    )

    ignored_keywords = {
        "define",
        "domain",
        "problem",
        ":init",
        ":goal",
        ":action",
        ":process",
        ":event",
        ":objects",
        ":metric",
    }

    for raw_line in planner_output.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        lower_line = line.lower()

        if any(keyword in lower_line for keyword in ignored_keywords):
            continue

        if timed_action_pattern.match(line) or plain_action_pattern.match(line):
            plan_lines.append(line)

    return plan_lines


def save_results_csv(results: list[dict[str, Any]], config: dict[str, Any]) -> Path:
    """Save planner execution results as CSV."""
    results_dir = ensure_directory(config["outputs"]["results_dir"])
    output_path = results_dir / "planner_results.csv"

    fieldnames = [
        "instance_name",
        "status",
        "return_code",
        "runtime_seconds",
        "plan_found",
        "plan_length",
        "problem_file",
        "plan_file",
        "log_file",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow({field: result.get(field) for field in fieldnames})

    print(f"\nSaved planner CSV results to: {output_path}")

    return output_path


def save_results_json(results: list[dict[str, Any]], config: dict[str, Any]) -> Path:
    """Save planner execution results as JSON."""
    results_dir = ensure_directory(config["outputs"]["results_dir"])
    output_path = results_dir / "planner_results.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"Saved planner JSON results to: {output_path}")

    return output_path


def run_planner(config: dict[str, Any]) -> dict[str, Any]:
    """
    Complete Stage Five:
    - verify Java
    - verify ENHSP JAR
    - verify PDDL files
    - run planner on small, medium, large problems
    - save logs, plans, and result summaries
    """
    java_path = config["planner"].get("java_path", "java")
    jar_path = Path(config["planner"]["jar_path"])
    domain_file = Path(config["planning"]["domain_file"])

    print("Checking Java...")
    java_version = check_java_available(java_path)
    print(java_version.splitlines()[0])

    check_file_exists(jar_path, "ENHSP planner JAR")
    check_file_exists(domain_file, "PDDL domain file")

    problem_files = get_problem_files(config)

    results = []

    for problem_file in problem_files:
        result = run_single_problem(
            config=config,
            domain_file=domain_file,
            problem_file=problem_file,
        )
        results.append(result)

    csv_path = save_results_csv(results, config)
    json_path = save_results_json(results, config)

    return {
        "results": results,
        "csv_path": str(csv_path),
        "json_path": str(json_path),
    }