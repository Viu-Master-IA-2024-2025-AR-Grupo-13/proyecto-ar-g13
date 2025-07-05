import re
import sys
import numpy as np
import statistics as stats
from pathlib import Path
from typing import List, Tuple

def read_source() -> str:
    """Return the log text from stdin or the first CLI argument."""
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
    else:
        print("Usage: python tests_rewards.py <file>")
        print()
        print("Quieres parsear tests.txt?")
        print("Prueba: python tests_rewards.py tests.txt")
        sys.exit(1)

def parse_rewards(log_text: str):
    """Extract rewards for each TEST block."""
    test_rewards = {}
    current_test = None

    for line in log_text.splitlines():
        line = line.lstrip()
        header = re.match(r">+\s*TEST\s+(\d+)", line)
        if header:
            current_test = int(header.group(1))
            test_rewards[current_test] = []
            continue

        if current_test is not None:
            reward_match = re.search(r"reward:\s*([0-9.]+)", line)
            if reward_match:
                test_rewards[current_test].append(float(reward_match.group(1)))
    
    return test_rewards

def describe(rewards: List[float]) -> Tuple[float, float, float, str]:
    """Return (avg, min, max, mode_str) for a list of rewards."""
    arr = np.array(rewards, dtype=float)
    avg = arr.mean()
    min_val = arr.min()
    max_val = arr.max()
    modes = stats.multimode(arr)
    mode_str = ", ".join(f"{m:.3f}" for m in modes)
    return avg, min_val, max_val, mode_str

def main() -> None:
    log_text = read_source()
    test_rewards = parse_rewards(log_text)

    # Build a list of (test_num, avg, min, max, mode_str) tuples
    stats_list = [
        (test_num, *describe(rewards))
        for test_num, rewards in test_rewards.items()
    ]

    # Sort by average reward (ascending); set reverse=True for descending
    stats_list.sort(key=lambda x: x[1], reverse=False)

    print("Reward statistics per test (sorted by average)")
    print("---------------------------------------------")
    for test_num, avg, min_val, max_val, mode_str in stats_list:
        print(
            f"TEST {test_num}: "
            f"avg {avg:.3f}, "
            f"min {min_val:.3f}, "
            f"max {max_val:.3f}, "
            f"mode {mode_str}"
        )

if __name__ == "__main__":
    main()