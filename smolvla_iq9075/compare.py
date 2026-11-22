"""Compare LIBERO success counts on strictly matched episode pairs."""
import argparse
from collections import Counter
import json
from pathlib import Path


def read(path):
    rows = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        key = (row["suite"], row["task"], row["init"])
        if key in rows:
            raise ValueError(f"Duplicate episode {key} in {path}")
        rows[key] = row
    return rows


def compare(baseline, candidate, lineage=None):
    if not baseline or baseline.keys() != candidate.keys():
        raise ValueError("Both result files require the same nonempty episode set")
    suites = {}
    for key, base in baseline.items():
        edge = candidate[key]
        if lineage is None:
            hashes = (base["checkpoint_sha256"], base["checkpoint_sha256"])
        else:
            hashes = (lineage["teacher_checkpoint_sha256"], lineage["student_checkpoint_sha256"])
            for row, field in ((base, "teacher_steps"), (edge, "student_steps")):
                if row.get("denoising_steps", lineage[field]) != lineage[field]:
                    raise ValueError(f"Mismatched denoising_steps in episode {key}")
        if (base["checkpoint_sha256"], edge["checkpoint_sha256"]) != hashes:
            raise ValueError(f"Mismatched checkpoint_sha256 in episode {key}")
        for field in ("seed", "initial_sha256", "execute", "bddl_sha256", "init_states_sha256"):
            if base[field] != edge[field]:
                raise ValueError(f"Mismatched {field} in episode {key}")
        label = {(True, True): "both_success", (True, False): "baseline_only",
                 (False, True): "candidate_only", (False, False): "both_failed"}
        suites.setdefault(key[0], Counter())[label[base["success"], edge["success"]]] += 1
    suites["all"] = sum(suites.values(), start=Counter())
    result = {}
    for suite, counters in suites.items():
        total = sum(counters.values())
        row = {key: counters[key] for key in ("both_success", "baseline_only", "candidate_only", "both_failed")}
        result[suite] = {
            "episodes": total, "baseline_successes": counters["both_success"] + counters["baseline_only"],
            "candidate_successes": counters["both_success"] + counters["candidate_only"],
            "delta_percentage_points": round(100 * (counters["candidate_only"] - counters["baseline_only"]) / total, 4),
            **row,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--distillation", type=Path, help="Teacher/student lineage from distill.py")
    args = parser.parse_args()
    lineage = json.loads(args.distillation.read_text()) if args.distillation else None
    print(json.dumps(compare(read(args.baseline), read(args.candidate), lineage), indent=2))


if __name__ == "__main__":
    main()
