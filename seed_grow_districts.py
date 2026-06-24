from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
import heapq
from pathlib import Path
import sys

import numpy as np

try:
    from scipy.spatial import cKDTree
except ModuleNotFoundError:
    cKDTree = None


OUTPUT_COLUMNS = [
    "number", "republican_number", "democratic_number", "longitude", "latitude",
    "population", "state_fips", "district", "district_target_party", "district_winner",
]


def parse_number(value: str) -> Decimal:
    try:
        number = Decimal(value.strip().replace(",", ""))
    except InvalidOperation as error:
        raise ValueError(f"invalid numeric value {value!r}") from error
    if not number.is_finite() or number < 0:
        raise ValueError(f"numbers must be finite and nonnegative, got {value!r}")
    return number


def prompt_path(prompt: str, default: Path | None = None) -> Path:
    suffix = f" [{default}]" if default else ""
    raw_value = input(f"{prompt}{suffix}: ").strip().strip('"')
    return Path(raw_value) if raw_value else Path(default)


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


def read_predictions(path: Path) -> dict[str, list[dict[str, object]]]:
    rows_by_state: dict[str, list[dict[str, object]]] = {}
    rows_used = 0
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        for line_number, row in enumerate(csv.reader(file), start=1):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) != 3:
                raise ValueError(f"{path.name} line {line_number} has {len(row)} columns; expected exactly 3")

            block_id = row[0].strip()
            try:
                republican = parse_number(row[1])
                democratic = parse_number(row[2])
            except ValueError:
                if rows_used == 0 and line_number == 1:
                    continue
                raise ValueError(f"{path.name} line {line_number} has an invalid prediction value") from None

            if not block_id.isdigit():
                raise ValueError(f"{path.name} line {line_number} has a nonnumeric census block ID")
            state_fips = block_id.zfill(15)[:2]
            rows_by_state.setdefault(state_fips, []).append(
                {"number": block_id, "republican": republican, "democratic": democratic}
            )
            rows_used += 1

    if rows_used == 0:
        raise ValueError(f"{path} contains no prediction rows")
    return rows_by_state


def coordinate_files_by_state(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        raise ValueError(f"coordinate folder not found: {folder}")
    matches: dict[str, Path] = {}
    for path in sorted(folder.glob("*.csv")):
        try:
            with path.open("r", newline="", encoding="utf-8-sig") as file:
                first = next(csv.reader(file), None)
        except OSError as error:
            raise ValueError(f"could not read {path}: {error}") from error
        if not first or not first[0].strip().isdigit():
            continue
        state_fips = first[0].strip().zfill(15)[:2]
        if state_fips in matches:
            raise ValueError(f"multiple coordinate CSVs found for state FIPS {state_fips}")
        matches[state_fips] = path
    return matches


def join_coordinates(prediction_rows: list[dict[str, object]], coordinate_path: Path) -> list[dict[str, object]]:
    predictions = {str(row["number"]): row for row in prediction_rows}
    joined: list[dict[str, object]] = []
    with coordinate_path.open("r", newline="", encoding="utf-8-sig") as file:
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
                raise ValueError(f"invalid coordinate row in {coordinate_path.name} line {line_number}") from error
            if not np.isfinite([longitude, latitude, population]).all() or population < 0:
                raise ValueError(f"invalid coordinate row in {coordinate_path.name} line {line_number}")
            joined.append(
                {
                    **prediction,
                    "longitude": longitude,
                    "latitude": latitude,
                    "population": population,
                }
            )

    missing = set(predictions) - {str(row["number"]) for row in joined}
    if missing:
        example = sorted(missing)[0]
        raise ValueError(
            f"{coordinate_path.name} is missing coordinates for {len(missing)} prediction rows "
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
        row, column = np.unravel_index(int(np.argmin(distances)), distances.shape)
        candidate = (
            float(np.sqrt(distances[row, column])),
            int(left_indexes[start + row]),
            int(right_indexes[column]),
        )
        if candidate < best:
            best = candidate
    return best


def build_adjacency(points: np.ndarray, neighbors: int) -> list[set[int]]:
    count = len(points)
    adjacency = [set() for _ in range(count)]
    if count == 1:
        return adjacency

    for left, row in enumerate(nearest_neighbors(points, neighbors)):
        for right in np.atleast_1d(row):
            right = int(right)
            adjacency[left].add(right)
            adjacency[right].add(left)

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
        party_margin = margin if party == "Republican" else -margin
        cutoff = np.quantile(party_margin, 0.60)
        candidates = np.flatnonzero(party_margin >= cutoff)
        candidates = candidates[np.isin(candidates, seeds, invert=True)]
        if len(candidates) == 0:
            candidates = np.asarray([index for index in range(len(points)) if index not in seeds])

        if len(seeds) == 0:
            weights = party_margin[candidates] - party_margin[candidates].min() + 0.01
            seed = int(rng.choice(candidates, p=weights / weights.sum()))
        else:
            distances = np.min(
                np.linalg.norm(
                    normalized[candidates, None, :] - normalized[np.asarray(seeds)][None, :, :],
                    axis=2,
                ),
                axis=1,
            )
            partisan = (party_margin[candidates] - party_margin.min()) / (np.ptp(party_margin) + 1e-9)
            seed = int(candidates[np.argmax(distances + 0.25 * partisan + rng.normal(0, 0.04, len(candidates)))])
        seeds.append(seed)
    return seeds


def grow_plan(
    points: np.ndarray,
    adjacency: list[set[int]],
    population: np.ndarray,
    margin: np.ndarray,
    target_parties: list[str],
    seeds: list[int],
    rng: np.random.Generator,
    compactness_weight: float,
    boundary_weight: float,
) -> np.ndarray:
    district_count = len(seeds)
    assignment = np.full(len(population), -1, dtype=int)
    district_population = np.zeros(district_count)
    frontiers: list[list[tuple[float, int]]] = [[] for _ in range(district_count)]
    ideal_population = max(float(population.sum()) / district_count, 1.0)
    scale = np.ptp(points, axis=0)
    scale[scale == 0] = 1
    normalized = (points - points.min(axis=0)) / scale
    seed_points = normalized[np.asarray(seeds)]
    margin_scale = max(float(np.std(margin)), 1.0)

    def add_frontier(district: int, node: int) -> None:
        if assignment[node] != -1:
            return
        sign = 1 if target_parties[district] == "Republican" else -1
        fullness = district_population[district] / ideal_population
        party_score = sign * margin[node] / margin_scale
        distance_from_seed = float(np.linalg.norm(normalized[node] - seed_points[district]))
        same_neighbors = 0
        other_neighbors = 0
        for neighbor in adjacency[node]:
            neighbor_assignment = assignment[neighbor]
            if neighbor_assignment == district:
                same_neighbors += 1
            elif neighbor_assignment != -1:
                other_neighbors += 1
        boundary_penalty = other_neighbors - same_neighbors * 0.35
        priority = (
            compactness_weight * distance_from_seed
            + boundary_weight * boundary_penalty
            - 0.25 * party_score
            + 4.0 * max(fullness - 0.95, 0.0)
            + rng.random() * 0.03
        )
        heapq.heappush(frontiers[district], (priority, node))

    for district, seed in enumerate(seeds):
        assignment[seed] = district
        district_population[district] = population[seed]
    for district, seed in enumerate(seeds):
        for neighbor in adjacency[seed]:
            add_frontier(district, neighbor)

    remaining = len(population) - district_count
    while remaining:
        district_order = np.argsort(district_population / ideal_population + rng.random(district_count) * 0.01)
        selected_district = selected_node = None
        for district in district_order:
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


def score_plan(
    assignment: np.ndarray,
    points: np.ndarray,
    adjacency: list[set[int]],
    population: np.ndarray,
    margin: np.ndarray,
    target_parties: list[str],
    desired_republican: int,
) -> tuple[float, int, float]:
    district_count = len(target_parties)
    district_population = np.bincount(assignment, weights=population, minlength=district_count)
    district_margin = np.bincount(assignment, weights=population * margin, minlength=district_count)
    republican_wins = int(np.sum(district_margin > 0))
    seat_error = abs(republican_wins - desired_republican)
    population_error = float(np.std(district_population) / (np.mean(district_population) + 1e-9))
    target_error = 0
    for district, target_party in enumerate(target_parties):
        winner = "Republican" if district_margin[district] > 0 else "Democratic"
        if winner != target_party:
            target_error += 1

    compactness_error = 0.0
    scale = np.ptp(points, axis=0)
    scale[scale == 0] = 1
    normalized = (points - points.min(axis=0)) / scale
    for district in range(district_count):
        members = np.flatnonzero(assignment == district)
        if len(members) == 0:
            compactness_error += 100.0
            continue
        centroid = np.average(normalized[members], axis=0, weights=population[members])
        compactness_error += float(np.average(np.linalg.norm(normalized[members] - centroid, axis=1), weights=population[members]))
    compactness_error /= district_count

    district_neighbors = [set() for _district in range(district_count)]
    boundary_edges = 0
    for node, neighbors in enumerate(adjacency):
        district = int(assignment[node])
        for neighbor in neighbors:
            other = int(assignment[neighbor])
            if other != district:
                district_neighbors[district].add(other)
                boundary_edges += 1
    tucked_inside_penalty = sum(1 for neighbors in district_neighbors if len(neighbors) == 1 and district_count > 2)
    boundary_error = boundary_edges / max(len(assignment), 1)

    return (
        seat_error * 1_000_000
        + target_error * 10_000
        + tucked_inside_penalty * 5_000
        + compactness_error * 1_000
        + boundary_error * 100
        + population_error,
        republican_wins,
        population_error,
    )


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
            raise RuntimeError(f"district {district + 1} is not connected")


def create_plan(
    rows: list[dict[str, object]],
    district_count: int,
    desired_republican: int,
    desired_democratic: int,
    attempts: int,
    random_seed: int,
    neighbors: int,
    compactness_weight: float,
    boundary_weight: float,
) -> tuple[np.ndarray, list[str], int, float]:
    points = np.asarray([[row["longitude"], row["latitude"]] for row in rows], dtype=float)
    population = np.maximum(np.asarray([row["population"] for row in rows], dtype=float), 1.0)
    margin = np.asarray([float(row["republican"]) - float(row["democratic"]) for row in rows], dtype=float)
    adjacency = build_adjacency(points, neighbors)
    rng = np.random.default_rng(random_seed)
    base_targets = ["Republican"] * desired_republican + ["Democratic"] * desired_democratic

    best_assignment = None
    best_targets = None
    best_score = (float("inf"), 0, 0.0)
    for attempt in range(attempts):
        targets = base_targets.copy()
        rng.shuffle(targets)
        seeds = choose_seeds(points, margin, targets, rng)
        assignment = grow_plan(
            points,
            adjacency,
            population,
            margin,
            targets,
            seeds,
            rng,
            compactness_weight,
            boundary_weight,
        )
        score = score_plan(assignment, points, adjacency, population, margin, targets, desired_republican)
        if score[0] < best_score[0]:
            best_assignment, best_targets, best_score = assignment, targets, score
        print(
            f"  candidate {attempt + 1}/{attempts}: "
            f"R {score[1]}, D {district_count - score[1]}, pop error {score[2]:.3f}",
            end="\r",
            flush=True,
        )
        if score[1] == desired_republican and score[0] < 10_000:
            break
    print()

    if best_assignment is None or best_targets is None:
        raise RuntimeError("no district plan was generated")
    validate_contiguity(adjacency, best_assignment)
    return best_assignment, best_targets, best_score[1], best_score[2]


def write_rows(
    writer: csv.DictWriter,
    rows: list[dict[str, object]],
    assignment: np.ndarray,
    target_parties: list[str],
    state_fips: str,
) -> None:
    district_count = len(target_parties)
    population = np.maximum(np.asarray([row["population"] for row in rows], dtype=float), 1.0)
    margin = np.asarray([float(row["republican"]) - float(row["democratic"]) for row in rows], dtype=float)
    district_margin = np.bincount(assignment, weights=population * margin, minlength=district_count)
    winners = ["Republican" if value > 0 else "Democratic" for value in district_margin]
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
                "district": district + 1,
                "district_target_party": target_parties[district],
                "district_winner": winners[district],
            }
        )


def prompt_targets(state_fips: str, row_count: int) -> tuple[int, int, int]:
    print(f"\nState FIPS {state_fips}: {row_count} prediction rows")
    district_count = prompt_integer("How many total districts do you want? ", 1, row_count)
    while True:
        republican = prompt_integer("How many Republican districts do you want? ", 0, district_count)
        democratic = prompt_integer("How many Democratic districts do you want? ", 0, district_count)
        if republican + democratic == district_count:
            return district_count, republican, democratic
        print(f"Those add up to {republican + democratic}, but you asked for {district_count} total districts.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_csv", nargs="?", type=Path, help="three-column prediction CSV")
    parser.add_argument("--coords-dir", type=Path, help="folder containing coordinate/population CSV files")
    parser.add_argument("--output", type=Path, help="output assignment CSV path")
    parser.add_argument("--attempts", type=int, default=50, help="candidate plans per state (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument("--neighbors", type=int, default=8, help="nearest neighbors used for contiguity graph")
    parser.add_argument(
        "--compactness-weight",
        type=float,
        default=3.0,
        help="higher values make districts grow closer to their seed (default: 3.0)",
    )
    parser.add_argument(
        "--boundary-weight",
        type=float,
        default=1.25,
        help="higher values discourage districts from wrapping around each other (default: 1.25)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.attempts < 1:
        raise ValueError("--attempts must be at least 1")
    if args.neighbors < 1:
        raise ValueError("--neighbors must be at least 1")
    if args.compactness_weight < 0:
        raise ValueError("--compactness-weight must be 0 or greater")
    if args.boundary_weight < 0:
        raise ValueError("--boundary-weight must be 0 or greater")

    prediction_path = args.prediction_csv or prompt_path("Prediction CSV file", Path("predict.csv"))
    coords_dir = args.coords_dir or prompt_path("Coordinate data folder", Path("testing data"))
    output_path = args.output or prediction_path.with_name(f"{prediction_path.stem}_seed_grow_districts.csv")

    predictions_by_state = read_predictions(prediction_path)
    coordinate_files = coordinate_files_by_state(coords_dir)
    choices = {
        state_fips: prompt_targets(state_fips, len(rows))
        for state_fips, rows in sorted(predictions_by_state.items())
    }

    summaries = []
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for offset, state_fips in enumerate(sorted(predictions_by_state)):
            if state_fips not in coordinate_files:
                raise ValueError(f"no coordinate CSV found for state FIPS {state_fips}")
            district_count, desired_republican, desired_democratic = choices[state_fips]
            print(f"\nState FIPS {state_fips}: loading {coordinate_files[state_fips].name}")
            joined = join_coordinates(predictions_by_state[state_fips], coordinate_files[state_fips])

            margin = np.asarray([float(row["republican"]) - float(row["democratic"]) for row in joined])
            if desired_democratic and not np.any(margin < 0):
                print("  Warning: no Democratic-leaning rows were found, so Democratic wins may be impossible.")
            if desired_republican and not np.any(margin > 0):
                print("  Warning: no Republican-leaning rows were found, so Republican wins may be impossible.")

            assignment, targets, actual_republican, population_error = create_plan(
                joined,
                district_count,
                desired_republican,
                desired_democratic,
                args.attempts,
                args.seed + offset,
                args.neighbors,
                args.compactness_weight,
                args.boundary_weight,
            )
            write_rows(writer, joined, assignment, targets, state_fips)
            summaries.append(
                (
                    state_fips,
                    district_count,
                    desired_republican,
                    desired_democratic,
                    actual_republican,
                    district_count - actual_republican,
                    population_error,
                )
            )

    print("\nState  Total  Desired R/D  Result R/D  Pop Error")
    print("-----  -----  -----------  ----------  ---------")
    for state_fips, total, desired_r, desired_d, actual_r, actual_d, population_error in summaries:
        print(
            f"{state_fips:5}  {total:5}  {desired_r:4}/{desired_d:<4}  "
            f"{actual_r:4}/{actual_d:<4}  {population_error:9.3f}"
        )
    print(f"\nAssignment CSV: {output_path}")
    print("Every district is connected in the nearest-neighbor graph.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, csv.Error) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2)
