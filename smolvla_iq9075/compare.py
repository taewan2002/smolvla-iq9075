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


def compare(baseline, candidate):
    if not baseline or baseline.keys() != candidate.keys():
        raise ValueError("Both result files require the same nonempty episode set")
    suites = {}
    for key, base in baseline.items():
        edge = candidate[key]
        for field in ("seed", "initial_sha256", "execute", "checkpoint_sha256",
                      "bddl_sha256", "init_states_sha256"):
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
    args = parser.parse_args()
    print(json.dumps(compare(read(args.baseline), read(args.candidate)), indent=2))


if __name__ == "__main__":
    main()
