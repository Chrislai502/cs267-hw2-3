import csv
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme()

# DATA_IN = "./update_particles.2.txt"
# DATA_OUT = "./update_particles.2.csv"
# TIMER_DATA_IN = "./update_particles.2.timers.txt"
# TIMER_DATA_OUT = "./update_particles.2.timers.csv"

DATA_IN = "./update_particles.3.txt"
DATA_OUT = "./update_particles.3.csv"
TIMER_DATA_IN = "./update_particles.3.timers.txt"
TIMER_DATA_OUT = "./update_particles.3.timers.csv"
STARTER_DATA_IN = "./starter.txt"
STARTER_DATA_OUT = "./starter.csv"

def parse_starter_data():
    with open(STARTER_DATA_IN, "r", encoding="utf-8") as f:
        data = f.readlines()

    data_by_particles = {}

    for line in data:
        match = re.search(
            r"Simulation Time = ([\.\d]+) seconds for (\d+) particles", line
        )
        if match is None:
            print("Unknown line", line)
            continue

        time_secs = float(match.group(1))
        num_particles = int(match.group(2))

        if num_particles not in data_by_particles:
            data_by_particles[num_particles] = []

        data_by_particles[num_particles].append(time_secs)

    # compute averages
    all_rows = []
    csv_rows = []
    for num_particles in sorted(data_by_particles):
        times = data_by_particles[num_particles]
        mean_time_secs = sum(times) / len(times)
        csv_rows.append((num_particles, mean_time_secs))
        all_rows.extend([[num_particles, time_secs] for time_secs in times])

    with open(STARTER_DATA_OUT, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["num_particles", "time_secs"])
        writer.writerows(csv_rows)

    arr = np.array(all_rows)
    plt.xlabel("# particles")
    plt.ylabel("Time (s)")
    plt.xscale("log")
    plt.yscale("log")
    sns.lineplot(x=arr[:, 0], y=arr[:, 1])
    plt.show()

def parse_data():
    with open(DATA_IN, "r", encoding="utf-8") as f:
        data = f.readlines()

    data_by_particles = {}

    for line in data:
        match = re.search(
            r"Simulation Time = ([\.\d]+) seconds for (\d+) particles", line
        )
        if match is None:
            print("Unknown line", line)
            continue

        time_secs = float(match.group(1))
        num_particles = int(match.group(2))

        if num_particles not in data_by_particles:
            data_by_particles[num_particles] = []

        data_by_particles[num_particles].append(time_secs)

    # compute averages
    all_rows = []
    csv_rows = []
    for num_particles in sorted(data_by_particles):
        times = data_by_particles[num_particles]
        mean_time_secs = sum(times) / len(times)
        csv_rows.append((num_particles, mean_time_secs))
        all_rows.extend([[num_particles, time_secs] for time_secs in times])

    with open(DATA_OUT, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["num_particles", "time_secs"])
        writer.writerows(csv_rows)

    arr = np.array(all_rows)
    plt.xlabel("# particles")
    plt.ylabel("Time (s)")
    plt.xscale("log")
    plt.yscale("log")
    sns.lineplot(x=arr[:, 0], y=arr[:, 1])
    plt.show()


def extract_float(s):
    match = re.search(r"[\.\d]+", s)
    assert match is not None
    return float(match.group())


def parse_timer_data():
    with open(TIMER_DATA_IN, "r", encoding="utf-8") as f:
        data = f.readlines()

    data_by_particles = {}
    cur_num_particles = -1
    cur_breakdown = {}
    for line in data:
        line = line.strip()
        if not line:
            # done with timer data
            assert cur_num_particles != -1
            if cur_num_particles not in data_by_particles:
                data_by_particles[cur_num_particles] = []
            data_by_particles[cur_num_particles].append(cur_breakdown)
            cur_breakdown = {}

        if "Compute forces" in line:
            cur_breakdown["compute_forces"] = extract_float(line)
        if "Move GPU" in line:
            cur_breakdown["move_gpu"] = extract_float(line)
        if "Count particles in bins" in line:
            cur_breakdown["count_particles"] = extract_float(line)
        if "Exclusive scan" in line:
            cur_breakdown["exclusive_scan"] = extract_float(line)
        if "Sort" in line:
            cur_breakdown["sort"] = extract_float(line)
        if "Simulation Time" in line:
            match = re.search(
                r"Simulation Time = ([\.\d]+) seconds for (\d+) particles", line
            )
            assert match is not None
            cur_num_particles = int(match.group(2))

    # compute averages
    csv_rows = []
    for num_particles in sorted(data_by_particles):
        breakdowns = data_by_particles[num_particles]
        breakdown_keys = breakdowns[0].keys()

        avg_breakdown = {}
        for key in breakdown_keys:
            total = 0
            for breakdown in breakdowns:
                total += breakdown[key]
            avg_breakdown[key] = total / len(breakdowns)

        avg_breakdown["num_particles"] = num_particles
        csv_rows.append(avg_breakdown)

    fieldnames = [
        "num_particles",
        "compute_forces",
        "move_gpu",
        "count_particles",
        "exclusive_scan",
        "sort",
    ]

    with open(TIMER_DATA_OUT, "w", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    arr = np.array([[row[key] for key in fieldnames] for row in csv_rows])
    plt.xscale("log")
    # plt.yscale("log")
    plt.xlabel("# particles")
    plt.ylabel("Time (s)")
    plt.stackplot(arr[:, 0], *arr[:, 1:].T, labels=fieldnames[1:])
    plt.show()

    totals = arr[:, 1:].sum(1)

    plt.xscale("log")
    plt.stackplot(arr[:, 0], *arr[:, 1:].T / totals, labels=fieldnames[1:])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parse_data()
    parse_starter_data()
    parse_timer_data()
