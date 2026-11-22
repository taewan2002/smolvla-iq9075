"""Summarize paired paced LIBERO episodes and chunk timing."""
import argparse
import json
from pathlib import Path

import numpy as np

from .compare import compare, read


def distribution(values):
    if not values:
        return {"samples": 0, "median": None, "p95": None}
    return {"samples": len(values), "median": round(float(np.median(values)), 4),
            "p95": round(float(np.percentile(values, 95)), 4)}


def summarize(rows):
    timings = {
        name: distribution([value for row in rows.values() for value in row[name]])
        for name in ("chunk_wait_ms", "inference_ms", "position_jumps", "rotation_jumps")
    }
    waits = [wait for row in rows.values() for wait in row["chunk_wait_ms"]]
    applied = sum(row["steps"] for row in rows.values())
    empty = sum(row["empty_ticks"] for row in rows.values())
    missed = sum(row["scheduler_missed_ticks"] for row in rows.values())
    return {
        "episodes": len(rows),
        "successes": sum(row["success"] for row in rows.values()),
        "applied_actions": applied,
        "empty_ticks": empty,
        "scheduler_missed_ticks": missed,
        "empty_tick_percent": round(100 * empty / (applied + empty), 4),
        "late_boundaries": sum(wait > 0 for wait in waits),
        "mean_chunk_wait_ms": round(sum(waits) / len(waits), 4) if waits else None,
        "control_seconds_per_episode": distribution([row["control_seconds"] for row in rows.values()]),
        "startup_ms": distribution([row["startup_ms"] for row in rows.values()]),
        **timings,
    }


def compare_pacing(sync, asynchronous, blended):
    settings = ("checkpoint_sha256", "execute", "control_hz", "predicted_actions", "denoising_steps", "backend")
    runs = {"sync": sync, "async": asynchronous, "async_blend": blended}
    profiles = {}
    if not sync:
        raise ValueError("The baseline must contain paced episodes")
    reference = next(iter(sync.values()))
    for mode, rows in runs.items():
        if any(row[key] != reference[key] for row in rows.values() for key in settings):
            raise ValueError(f"{mode} uses different model or timing settings")
        configurations = {(row["refill_actions"], row["overlap"]) for row in rows.values()}
        if len(configurations) != 1:
            raise ValueError(f"{mode} must use a single refill and overlap configuration")
        profiles[mode] = configurations.pop()
    sync_profile, async_profile, blend_profile = profiles.values()
    if not (sync_profile == (0, 0) and async_profile[0] > 0 and async_profile[1] == 0
            and blend_profile[0] == async_profile[0] and blend_profile[1] > 0):
        raise ValueError("Expected sync, async and async with blending")
    return {
        "settings": {key: reference[key] for key in settings},
        "profiles": {name: {"refill_actions": refill, "overlap": overlap}
                     for name, (refill, overlap) in profiles.items()},
        "paired_success": {"async_vs_sync": compare(sync, asynchronous),
                           "blend_vs_sync": compare(sync, blended),
                           "blend_vs_async": compare(asynchronous, blended)},
        "metrics": {name: summarize(rows) for name, rows in runs.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sync", type=Path)
    parser.add_argument("async_run", type=Path)
    parser.add_argument("blend", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare_pacing(read(args.sync), read(args.async_run), read(args.blend)), indent=2))


if __name__ == "__main__":
    main()
