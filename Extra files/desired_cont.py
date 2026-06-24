"""Create contiguous, user-targeted districts from a prediction CSV.

The prediction CSV contains block_id, Republican value, Democratic value.
Coordinates and population are joined by block ID from the matching state CSV
inside the Data Training folder. Contiguity is measured on a geographic
nearest-neighbor graph built from census-block centroid coordinates.
"""

from __future__ import annotations

import argparse
import csv
import heapq
from pathlib import Path
import sys

import numpy as np

try:
    from scipy.spatial import cKDTree
except ModuleNotFoundError:
    cKDTree = None

from districts import STATE_INFO, read_data


OUTPUT_COLUMNS = [
    "number", "republican_number", "democratic_number", "longitude", "latitude",
    "population", "state_fips", "state", "district", "district_target_party",
    "district_winner",
]


def prompt_integer(prompt: str, minimum: int, maximum: int) -> int:
    while True:
        raw_value = input(prompt).strip()
        try:
            value = int(raw_value)
        except ValueError:
            print(f"Please enter a whole number from {minimum} to {maximum}.")
            continue
        if minimum <= value <= maximum:
            return value
        print(f"Please enter a whole number from {minimum} to {maximum}.")


def prompt_party_counts(state_name: str, district_count: int) -> tuple[int, int]:
    print(f"\n{state_name} has {district_count} district(s).")
    while True:
        republican = prompt_integer(
            f"How many Republican districts do you want in {state_name}? ",
            0,
            district_count,
        )
        democratic = prompt_integer(
            f"How many Democratic districts do you want in {state_name}? ",
            0,
            district_count,
        )
        if republican + democratic == district_count:
            return republican, democratic
        print(
            f"Those add up to {republican + democratic}, but {state_name} needs "
            f"exactly {district_count} districts. Try again."
        )


def training_files_by_state(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        raise ValueError(f"training-data folder not found: {folder}")
    matches: dict[str, Path] = {}
    for path in sorted(folder.glob("*.csv")):
        try:
            with path.open("r", newline="", encoding="utf-8-sig") as file:
                first = next(csv.reader(file), None)
            if not first or not first[0].strip().isdigit():
                continue
            state_fips = first[0].strip().zfill(15)[:2]
            if state_fips in matches:
                raise ValueError(f"multiple training CSVs found for state FIPS {state_fips}")
            matches[state_fips] = path
        except OSError as error:
            raise ValueError(f"could not read {path}: {error}") from error
    return matches


def join_coordinates(
    prediction_rows: list[dict[str, object]], training_path: Path
) -> list[dict[str, object]]:
    predictions = {str(row["number"]): row for row in prediction_rows}
    joined: list[dict[str, object]] = []
    with training_path.open("r", newline="", encoding="utf-8-sig") as file:
        for line_number, row in enumerate(csv.reader(file), start=1):
            if len(row) < 4:
                continue
            block_id = row[0].strip()
            prediction = predictions.get(block_id)
            if prediction is None:
                continue
            try:
                longitude = float(row[1])
                latitude = float(row[2])
                population = float(row[3])
            except ValueError as error:
                raise ValueError(f"invalid coordinates or population on {training_path.name} line {line_number}") from error
            if not np.isfinite([longitude, latitude, population]).all() or population < 0:
                raise ValueError(f"invalid coordinates or population on {training_path.name} line {line_number}")
            joined.append(
                {
                    **prediction,
                    "longitude": longitude,
                    "latitude": latitude,
                    "population": population,
                }
            )

    found = {str(row["number"]) for row in joined}
    missing = set(predictions) - found
    if missing:
        example = sorted(missing)[0]
        raise ValueError(
            f"{training_path.name} is missing coordinates for {len(missing)} prediction rows "
            f"(example block ID: {example})"
        )
    return joined


def graph_components(adjacency: list[set[int]]) -> list[np.ndarray]:
    unseen = set(range(len(adjacency)))
    components = []
    while unseen:
        stack = [unseen.pop()]
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            neighbors = adjacency[node] & unseen
            unseen.difference_update(neighbors)
            stack.extend(neighbors)
        components.append(np.asarray(component, dtype=int))
    return components


def nearest_neighbors(points: np.ndarray, neighbors: int) -> np.ndarray:
    count = len(points)
    neighbor_count = min(neighbors, count - 1)
    if neighbor_count < 1:
        return np.empty((count, 0), dtype=int)
    if cKDTree is not None:
        _, nearest = cKDTree(points).query(points, k=neighbor_count + 1)
        if nearest.ndim == 1:
            nearest = nearest[:, None]
        return np.asarray(nearest[:, 1:], dtype=int)

    indexes = np.empty((count, neighbor_count), dtype=int)
    chunk_size = max(1, min(count, 4_000_000 // max(count, 1)))
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        distances = np.sum((points[start:end, None, :] - points[None, :, :]) ** 2, axis=2)
        distances[np.arange(end - start), np.arange(start, end)] = np.inf
        partitioned = np.argpartition(distances, kth=neighbor_count - 1, axis=1)[:, :neighbor_count]
        indexes[start:end] = np.take_along_axis(
            partitioned,
            np.argsort(np.take_along_axis(distances, partitioned, axis=1), axis=1),
            axis=1,
        )
    return indexes


def closest_between(points: np.ndarray, left_indexes: np.ndarray, right_indexes: np.ndarray) -> tuple[float, int, int]:
    if cKDTree is not None:
        distances, indexes = cKDTree(points[right_indexes]).query(points[left_indexes], k=1)
        position = int(np.argmin(distances))
        return (
            float(distances[position]),
            int(left_indexes[position]),
            int(right_indexes[int(indexes[position])]),
        )

    best = (float("inf"), int(left_indexes[0]), int(right_indexes[0]))
    chunk_size = max(1, min(len(left_indexes), 4_000_000 // max(len(right_indexes), 1)))
    for start in range(0, len(left_indexes), chunk_size):
        end = min(start + chunk_size, len(left_indexes))
        distances = np.sum(
            (points[left_indexes[start:end], None, :] - points[right_indexes][None, :, :]) ** 2,
            axis=2,
        )
        flat_position = int(np.argmin(distances))
        row, column = np.unravel_index(flat_position, distances.shape)
        candidate = (
            float(np.sqrt(distances[row, column])),
            int(left_indexes[start + row]),
            int(right_indexes[column]),
        )
        if candidate < best:
            best = candidate
    return best


def build_adjacency(points: np.ndarray, neighbors: int = 6) -> list[set[int]]:
    count = len(points)
    adjacency = [set() for _ in range(count)]
    if count == 1:
        return adjacency
    nearest = nearest_neighbors(points, neighbors)
    for left, row in enumerate(nearest):
        for right in np.atleast_1d(row):
            right = int(right)
            if right != left:
                adjacency[left].add(right)
                adjacency[right].add(left)

    # A state may contain islands or isolated coordinate clusters. Join graph
    # components by their closest centroids so district growth can cover all rows.
    components = graph_components(adjacency)
    while len(components) > 1:
        largest = max(components, key=len)
        best = None
        for component in components:
            if component is largest:
                continue
            candidate = closest_between(points, component, largest)
            if best is None or candidate < best:
                best = candidate
        _, left, right = best
        adjacency[left].add(right)
        adjacency[right].add(left)
        components = graph_components(adjacency)
    return adjacency


def choose_seeds(
    points: np.ndarray,
    margin: np.ndarray,
    target_parties: list[str],
    rng: np.random.Generator,
) -> list[int]:
    scale = np.ptp(points, axis=0)
    scale[scale == 0] = 1
    normalized = (points - points.min(axis=0)) / scale
    seeds: list[int] = []
    for party in target_parties:
        desired_margin = margin if party == "Republican" else -margin
        cutoff = np.quantile(desired_margin, 0.55)
        candidates = np.flatnonzero(desired_margin >= cutoff)
        candidates = candidates[np.isin(candidates, seeds, invert=True)]
        if not len(candidates):
            candidates = np.asarray([index for index in range(len(points)) if index not in seeds])
        if not seeds:
            weights = desired_margin[candidates] - desired_margin[candidates].min() + 0.01
            seed = int(rng.choice(candidates, p=weights / weights.sum()))
        else:
            distances = np.min(
                np.linalg.norm(
                    normalized[candidates, None, :] - normalized[np.asarray(seeds)][None, :, :], axis=2
                ),
                axis=1,
            )
            partisan = (desired_margin[candidates] - desired_margin.min()) / (np.ptp(desired_margin) + 1e-9)
            scores = distances + 0.15 * partisan + rng.normal(0, 0.03, len(candidates))
            seed = int(candidates[np.argmax(scores)])
        seeds.append(seed)
    return seeds


def grow_plan(
    adjacency: list[set[int]],
    population: np.ndarray,
    margin: np.ndarray,
    target_parties: list[str],
    seeds: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    district_count = len(seeds)
    assignment = np.full(len(population), -1, dtype=int)
    district_population = np.zeros(district_count)
    frontiers: list[list[tuple[float, int]]] = [[] for _ in range(district_count)]
    ideal_population = max(float(population.sum()) / district_count, 1.0)

    def add_frontier(district: int, node: int) -> None:
        if assignment[node] != -1:
            return
        sign = 1 if target_parties[district] == "Republican" else -1
        priority = -(sign * margin[node]) + rng.random() * 0.08
        heapq.heappush(frontiers[district], (priority, node))

    for district, seed in enumerate(seeds):
        assignment[seed] = district
        district_population[district] = population[seed]
    for district, seed in enumerate(seeds):
        for neighbor in adjacency[seed]:
            add_frontier(district, neighbor)

    remaining = len(population) - district_count
    while remaining:
        order = np.argsort(district_population / ideal_population + rng.random(district_count) * 0.01)
        selected_district = selected_node = None
        for district in order:
            while frontiers[district] and assignment[frontiers[district][0][1]] != -1:
                heapq.heappop(frontiers[district])
            if frontiers[district]:
                _, selected_node = heapq.heappop(frontiers[district])
                selected_district = int(district)
                break
        if selected_node is None:
            raise RuntimeError("could not complete a connected district plan")
        assignment[selected_node] = selected_district
        district_population[selected_district] += population[selected_node]
        remaining -= 1
        for neighbor in adjacency[selected_node]:
            add_frontier(selected_district, neighbor)
    return assignment


def plan_score(
    assignment: np.ndarray,
    population: np.ndarray,
    margin: np.ndarray,
    desired_republican: int,
) -> tuple[float, int]:
    district_count = int(assignment.max()) + 1
    district_population = np.bincount(assignment, weights=population, minlength=district_count)
    district_margin = np.bincount(assignment, weights=population * margin, minlength=district_count)
    republican_wins = int(np.sum(district_margin > 0))
    population_error = float(np.std(district_population) / (np.mean(district_population) + 1e-9))
    return abs(republican_wins - desired_republican) * 1000 + population_error, republican_wins


def validate_contiguity(adjacency: list[set[int]], assignment: np.ndarray) -> None:
    for district in range(int(assignment.max()) + 1):
        members = set(np.flatnonzero(assignment == district).tolist())
        if not members:
            raise RuntimeError(f"district {district + 1} is empty")
        unseen = members.copy()
        stack = [unseen.pop()]
        while stack:
            node = stack.pop()
            connected = adjacency[node] & unseen
            unseen.difference_update(connected)
            stack.extend(connected)
        if unseen:
            raise RuntimeError(f"district {district + 1} is not geographically connected")


def create_plan(
    rows: list[dict[str, object]],
    desired_republican: int,
    desired_democratic: int,
    attempts: int,
    random_seed: int,
) -> tuple[np.ndarray, list[str], int]:
    points = np.asarray([[row["longitude"], row["latitude"]] for row in rows], dtype=float)
    population = np.maximum(np.asarray([row["population"] for row in rows], dtype=float), 1.0)
    margin = np.asarray(
        [float(row["republican"]) - float(row["democratic"]) for row in rows], dtype=float
    )
    adjacency = build_adjacency(points)
    rng = np.random.default_rng(random_seed)
    base_targets = ["Republican"] * desired_republican + ["Democratic"] * desired_democratic
    best_assignment = None
    best_targets = None
    best_score = (float("inf"), 0)

    for attempt in range(attempts):
        targets = base_targets.copy()
        rng.shuffle(targets)
        seeds = choose_seeds(points, margin, targets, rng)
        assignment = grow_plan(adjacency, population, margin, targets, seeds, rng)
        score = plan_score(assignment, population, margin, desired_republican)
        if score[0] < best_score[0]:
            best_assignment, best_targets, best_score = assignment, targets, score
        print(
            f"  candidate {attempt + 1}/{attempts}: Republican {score[1]}, "
            f"Democratic {len(targets) - score[1]}",
            end="\r",
            flush=True,
        )
        if score[1] == desired_republican and score[0] < 1000:
            break
    print()
    if best_assignment is None or best_targets is None:
        raise RuntimeError("no district plan was generated")
    validate_contiguity(adjacency, best_assignment)
    return best_assignment, best_targets, best_score[1]


def write_state_rows(
    writer: csv.DictWriter,
    rows: list[dict[str, object]],
    assignment: np.ndarray,
    target_parties: list[str],
    state_fips: str,
) -> None:
    district_count = len(target_parties)
    population = np.maximum(np.asarray([row["population"] for row in rows], dtype=float), 1.0)
    margin = np.asarray(
        [float(row["republican"]) - float(row["democratic"]) for row in rows], dtype=float
    )
    district_margin = np.bincount(assignment, weights=population * margin, minlength=district_count)
    winners = ["Republican" if value > 0 else "Democratic" for value in district_margin]
    state_name = STATE_INFO[state_fips][0]
    order = np.lexsort((np.asarray([str(row["number"]) for row in rows]), assignment))
    for index in order:
        row = rows[int(index)]
        district = int(assignment[index])
        writer.writerow(
            {
                "number": row["number"],
                "republican_number": row["republican"],
                "democratic_number": row["democratic"],
                "longitude": row["longitude"],
                "latitude": row["latitude"],
                "population": row["population"],
                "state_fips": state_fips,
                "state": state_name,
                "district": district + 1,
                "district_target_party": target_parties[district],
                "district_winner": winners[district],
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", nargs="?", type=Path, help="three-column prediction CSV")
    parser.add_argument("--training-dir", type=Path, default=Path(__file__).resolve().parent / "Data Training")
    parser.add_argument("--output", type=Path, help="output CSV path")
    parser.add_argument("--attempts", type=int, default=20, help="candidate plans per state (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.attempts < 1:
        raise ValueError("--attempts must be at least 1")
    input_path = args.csv_file or Path(input("Three-column prediction CSV file: ").strip().strip('"'))
    totals, prediction_rows = read_data(input_path)
    available_training = training_files_by_state(args.training_dir)
    choices: dict[str, tuple[int, int]] = {}

    print(f"Found {len(totals)} state(s) in {input_path.name}.")
    for state_fips in sorted(totals):
        if state_fips not in available_training:
            raise ValueError(
                f"no Data Training CSV found for {STATE_INFO[state_fips][0]} (FIPS {state_fips})"
            )
        choices[state_fips] = prompt_party_counts(STATE_INFO[state_fips][0], STATE_INFO[state_fips][1])

    output_path = args.output or input_path.with_name(f"{input_path.stem}_desired_contiguous_assignments.csv")
    summaries = []
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for offset, state_fips in enumerate(sorted(totals)):
            state_name = STATE_INFO[state_fips][0]
            print(f"\nLoading coordinates for {state_name}: {available_training[state_fips].name}")
            joined = join_coordinates(prediction_rows[state_fips], available_training[state_fips])
            desired_republican, desired_democratic = choices[state_fips]
            assignment, targets, actual_republican = create_plan(
                joined,
                desired_republican,
                desired_democratic,
                args.attempts,
                args.seed + offset,
            )
            write_state_rows(writer, joined, assignment, targets, state_fips)
            summaries.append(
                (state_name, len(targets), desired_republican, desired_democratic,
                 actual_republican, len(targets) - actual_republican)
            )

    print("\nState              Total  Desired R/D  Result R/D")
    print("-----------------  -----  -----------  ----------")
    for state, total, desired_r, desired_d, actual_r, actual_d in summaries:
        print(f"{state:17}  {total:5}  {desired_r:4}/{desired_d:<4}  {actual_r:4}/{actual_d:<4}")
    print(f"\nAssignment CSV: {output_path}")
    print("Every district is connected in the geographic nearest-neighbor graph.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, csv.Error) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2)
