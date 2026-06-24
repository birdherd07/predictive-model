"""Estimate a state's Republican and Democratic district totals from a CSV.

Input rows contain: census_block_id, republican_number, democratic_number.
The file may be headerless or have one header row.
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path
import sys


# 2020 Census apportionment determines voting U.S. House seats for 2022-2030.
STATE_INFO = {
    "01": ("Alabama", 7), "02": ("Alaska", 1), "04": ("Arizona", 9),
    "05": ("Arkansas", 4), "06": ("California", 52), "08": ("Colorado", 8),
    "09": ("Connecticut", 5), "10": ("Delaware", 1), "12": ("Florida", 28),
    "13": ("Georgia", 14), "15": ("Hawaii", 2), "16": ("Idaho", 2),
    "17": ("Illinois", 17), "18": ("Indiana", 9), "19": ("Iowa", 4),
    "20": ("Kansas", 4), "21": ("Kentucky", 6), "22": ("Louisiana", 6),
    "23": ("Maine", 2), "24": ("Maryland", 8), "25": ("Massachusetts", 9),
    "26": ("Michigan", 13), "27": ("Minnesota", 8), "28": ("Mississippi", 4),
    "29": ("Missouri", 8), "30": ("Montana", 2), "31": ("Nebraska", 3),
    "32": ("Nevada", 4), "33": ("New Hampshire", 2), "34": ("New Jersey", 12),
    "35": ("New Mexico", 3), "36": ("New York", 26), "37": ("North Carolina", 14),
    "38": ("North Dakota", 1), "39": ("Ohio", 15), "40": ("Oklahoma", 5),
    "41": ("Oregon", 6), "42": ("Pennsylvania", 17), "44": ("Rhode Island", 2),
    "45": ("South Carolina", 7), "46": ("South Dakota", 1), "47": ("Tennessee", 9),
    "48": ("Texas", 38), "49": ("Utah", 4), "50": ("Vermont", 1),
    "51": ("Virginia", 11), "53": ("Washington", 10), "54": ("West Virginia", 2),
    "55": ("Wisconsin", 8), "56": ("Wyoming", 1),
}


def parse_number(value: str) -> Decimal:
    try:
        number = Decimal(value.strip().replace(",", ""))
    except InvalidOperation as error:
        raise ValueError(f"invalid numeric value {value!r}") from error
    if not number.is_finite() or number < 0:
        raise ValueError(f"numbers must be finite and nonnegative, got {value!r}")
    return number


def read_data(
    path: Path,
) -> tuple[dict[str, dict[str, Decimal | int]], dict[str, list[dict[str, str | Decimal]]]]:
    totals: dict[str, dict[str, Decimal | int]] = {}
    records: dict[str, list[dict[str, str | Decimal]]] = {}
    rows_used = 0

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        for line_number, row in enumerate(reader, start=1):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) != 3:
                raise ValueError(f"line {line_number} has {len(row)} columns; expected exactly 3")

            block_id = row[0].strip()
            try:
                republican = parse_number(row[1])
                democratic = parse_number(row[2])
            except ValueError:
                if rows_used == 0 and line_number == 1:
                    continue  # Optional header row.
                raise ValueError(f"line {line_number} has an invalid party number") from None

            if not block_id.isdigit():
                raise ValueError(f"line {line_number} has a nonnumeric census block ID")
            padded_id = block_id.zfill(15)
            row_state = padded_id[:2]
            if row_state not in STATE_INFO:
                raise ValueError(f"line {line_number} has unsupported state FIPS {row_state}")
            state_totals = totals.setdefault(
                row_state,
                {"rows": 0, "republican": Decimal(0), "democratic": Decimal(0)},
            )
            state_totals["rows"] += 1
            state_totals["republican"] += republican
            state_totals["democratic"] += democratic
            records.setdefault(row_state, []).append(
                {"number": block_id, "republican": republican, "democratic": democratic}
            )
            rows_used += 1

    if rows_used == 0:
        raise ValueError("the CSV contains no valid data rows")
    return totals, records


def estimate_districts(
    seats: int, republican_total: Decimal, democratic_total: Decimal
) -> tuple[int, int, Decimal, Decimal]:
    two_party_total = republican_total + democratic_total
    if two_party_total == 0:
        raise ValueError("the Republican and Democratic totals are both zero")

    republican_share = republican_total / two_party_total
    democratic_share = democratic_total / two_party_total
    # Round half up, then give the remainder to the other party so totals always match.
    republican_seats = int(republican_share * seats + Decimal("0.5"))
    democratic_seats = seats - republican_seats
    return republican_seats, democratic_seats, republican_share, democratic_share


def assign_districts(
    records: list[dict[str, str | Decimal]], republican_seats: int, democratic_seats: int
) -> list[dict[str, str | Decimal | int]]:
    """Assign rows to block-count-balanced, party-targeted districts."""
    district_parties = ["Republican"] * republican_seats + ["Democratic"] * democratic_seats
    district_count = len(district_parties)
    base_size, extra = divmod(len(records), district_count)
    capacities = [base_size + (district < extra) for district in range(district_count)]

    ordered = sorted(
        records,
        key=lambda row: (row["republican"] - row["democratic"], row["number"]),
        reverse=True,
    )
    republican_capacity = sum(capacities[:republican_seats])
    party_pools = [ordered[:republican_capacity], ordered[republican_capacity:]]
    assignments: list[dict[str, str | Decimal | int]] = []

    for district_indexes, pool in (
        (list(range(republican_seats)), party_pools[0]),
        (list(range(republican_seats, district_count)), party_pools[1]),
    ):
        cursor = 0
        for district in district_indexes:
            size = capacities[district]
            district_rows = pool[cursor : cursor + size]
            cursor += size
            for row in district_rows:
                assignments.append(
                    {
                        **row,
                        "district": district + 1,
                        "district_target_party": district_parties[district],
                    }
                )
    return assignments


def write_assignments(
    path: Path,
    records_by_state: dict[str, list[dict[str, str | Decimal]]],
    results: list[dict[str, object]],
) -> int:
    result_by_fips = {str(result["fips"]): result for result in results}
    rows_written = 0
    with path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "number", "republican_number", "democratic_number", "state_fips",
            "state", "district", "district_target_party", "district_winner",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for state_fips in sorted(records_by_state):
            result = result_by_fips[state_fips]
            assignments = assign_districts(
                records_by_state[state_fips],
                int(result["republican_seats"]),
                int(result["democratic_seats"]),
            )
            district_totals: dict[int, list[Decimal]] = {}
            for assignment in assignments:
                district = int(assignment["district"])
                values = district_totals.setdefault(district, [Decimal(0), Decimal(0)])
                values[0] += Decimal(assignment["republican"])
                values[1] += Decimal(assignment["democratic"])
            winners = {
                district: "Republican" if values[0] > values[1] else "Democratic"
                for district, values in district_totals.items()
            }
            for assignment in sorted(assignments, key=lambda row: (row["district"], row["number"])):
                writer.writerow(
                    {
                        "number": assignment["number"],
                        "republican_number": assignment["republican"],
                        "democratic_number": assignment["democratic"],
                        "state_fips": state_fips,
                        "state": result["state"],
                        "district": assignment["district"],
                        "district_target_party": assignment["district_target_party"],
                        "district_winner": winners[int(assignment["district"])],
                    }
                )
                rows_written += 1
    return rows_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", nargs="?", type=Path, help="three-column prediction or vote CSV")
    parser.add_argument("--output", type=Path, help="assignment CSV path (default: based on input name)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = args.csv_file or Path(input("Three-column CSV file: ").strip().strip('"'))
    totals, records_by_state = read_data(path)
    results = []
    for state_fips in sorted(totals):
        state_name, districts = STATE_INFO[state_fips]
        values = totals[state_fips]
        republican_total = values["republican"]
        democratic_total = values["democratic"]
        republican_seats, democratic_seats, republican_share, democratic_share = estimate_districts(
            districts, republican_total, democratic_total
        )
        results.append(
            {
                "fips": state_fips,
                "state": state_name,
                "rows": values["rows"],
                "districts": districts,
                "republican_total": republican_total,
                "democratic_total": democratic_total,
                "republican_share": republican_share,
                "democratic_share": democratic_share,
                "republican_seats": republican_seats,
                "democratic_seats": democratic_seats,
            }
        )

    headers = ["FIPS", "State", "Rows", "Districts", "R total", "D total", "R share", "D share", "R seats", "D seats"]
    table_rows = [
        [
            result["fips"], result["state"], str(result["rows"]), str(result["districts"]),
            f'{result["republican_total"]:.2f}', f'{result["democratic_total"]:.2f}',
            f'{result["republican_share"]:.2%}', f'{result["democratic_share"]:.2%}',
            str(result["republican_seats"]), str(result["democratic_seats"]),
        ]
        for result in results
    ]
    widths = [max(len(headers[index]), *(len(row[index]) for row in table_rows)) for index in range(len(headers))]

    def print_row(row: list[str]) -> None:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))

    print_row(headers)
    print_row(["-" * width for width in widths])
    for row in table_rows:
        print_row(row)

    print()
    print(f"States: {len(results)}")
    print(f"Rows used: {sum(int(result['rows']) for result in results)}")
    print(f"Total districts: {sum(int(result['districts']) for result in results)}")
    print(f"Republican districts: {sum(int(result['republican_seats']) for result in results)}")
    print(f"Democratic districts: {sum(int(result['democratic_seats']) for result in results)}")
    output_path = args.output or path.with_name(f"{path.stem}_district_assignments.csv")
    rows_written = write_assignments(output_path, records_by_state, results)
    print(f"Assignment CSV: {output_path} ({rows_written} rows)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, csv.Error) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2)
