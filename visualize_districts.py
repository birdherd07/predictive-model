"""Draw district assignment maps from a generated district CSV.

The input CSV should include longitude, latitude, state_fips, and district
columns, such as the output from seed_grow_districts.py.
"""

from __future__ import annotations

import argparse
import colorsys
import csv
from pathlib import Path
import random
import sys

from PIL import Image, ImageDraw, ImageFont


REQUIRED_COLUMNS = {"longitude", "latitude", "state_fips", "district"}


def prompt_path(prompt: str, default: Path | None = None) -> Path:
    suffix = f" [{default}]" if default else ""
    raw_value = input(f"{prompt}{suffix}: ").strip().strip('"')
    if raw_value:
        return Path(raw_value)
    if default is None:
        raise ValueError(f"{prompt} is required")
    return default


def read_assignments(path: Path) -> dict[str, list[tuple[float, float, int]]]:
    states: dict[str, list[tuple[float, float, int]]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing required column(s): {', '.join(sorted(missing))}")

        for line_number, row in enumerate(reader, start=2):
            try:
                longitude = float(row["longitude"])
                latitude = float(row["latitude"])
                district = int(row["district"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid map row on line {line_number}") from error
            state_fips = row["state_fips"].strip().zfill(2)
            states.setdefault(state_fips, []).append((longitude, latitude, district))

    if not states:
        raise ValueError(f"{path} contains no assignment rows")
    return states


def district_colors(districts: list[int], seed: int) -> dict[int, tuple[int, int, int]]:
    rng = random.Random(seed)
    hues = [index / max(len(districts), 1) for index in range(len(districts))]
    rng.shuffle(hues)

    colors = {}
    for district, hue in zip(sorted(districts), hues):
        hue = (hue + rng.uniform(-0.035, 0.035)) % 1.0
        saturation = rng.uniform(0.55, 0.85)
        value = rng.uniform(0.65, 0.95)
        red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
        colors[district] = (int(red * 255), int(green * 255), int(blue * 255))
    return colors


def fit_points(
    rows: list[tuple[float, float, int]],
    width: int,
    height: int,
    left_margin: int,
    top_margin: int,
    right_margin: int,
    bottom_margin: int,
) -> list[tuple[int, int, int]]:
    longitudes = [row[0] for row in rows]
    latitudes = [row[1] for row in rows]
    lon_min, lon_max = min(longitudes), max(longitudes)
    lat_min, lat_max = min(latitudes), max(latitudes)
    lon_range = max(lon_max - lon_min, 1e-9)
    lat_range = max(lat_max - lat_min, 1e-9)

    draw_width = width - left_margin - right_margin
    draw_height = height - top_margin - bottom_margin
    scale = min(draw_width / lon_range, draw_height / lat_range)
    used_width = lon_range * scale
    used_height = lat_range * scale
    x_offset = left_margin + (draw_width - used_width) / 2
    y_offset = bottom_margin + (draw_height - used_height) / 2

    points = []
    for longitude, latitude, district in rows:
        x = int(round((longitude - lon_min) * scale + x_offset))
        y = int(round(height - ((latitude - lat_min) * scale + y_offset)))
        points.append((x, y, district))
    return points


def draw_square(draw: ImageDraw.ImageDraw, x: int, y: int, radius: int, color: tuple[int, int, int]) -> None:
    if radius <= 0:
        draw.point((x, y), fill=color)
    else:
        draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=color)


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_legend(
    image: Image.Image,
    colors: dict[int, tuple[int, int, int]],
    title: str,
) -> None:
    draw = ImageDraw.Draw(image)
    title_font = load_font(28)
    label_font = load_font(18)
    draw.text((24, 18), title, fill=(20, 20, 20), font=title_font)

    legend_x = 24
    legend_y = 58
    box = 15
    gap = 8
    districts = sorted(colors)
    max_items = min(len(districts), 36)
    for offset, district in enumerate(districts[:max_items]):
        column = offset // 18
        row = offset % 18
        x = legend_x + column * 130
        y = legend_y + row * 22
        draw.rectangle((x, y, x + box, y + box), fill=colors[district])
        draw.text((x + box + gap, y - 2), f"District {district}", fill=(30, 30, 30), font=label_font)
    if len(districts) > max_items:
        draw.text(
            (legend_x, legend_y + 18 * 22 + 8),
            f"+ {len(districts) - max_items} more districts",
            fill=(30, 30, 30),
            font=label_font,
        )


def legend_height(district_count: int) -> int:
    rows = min(18, district_count)
    extra = 28 if district_count > 36 else 0
    return 58 + rows * 22 + extra + 28


def render_state(
    state_fips: str,
    rows: list[tuple[float, float, int]],
    output_dir: Path,
    width: int,
    height: int,
    point_radius: int,
    seed: int,
) -> Path:
    districts = sorted({district for _longitude, _latitude, district in rows})
    colors = district_colors(districts, seed + int(state_fips))
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    points = fit_points(
        rows,
        width,
        height,
        left_margin=80,
        top_margin=legend_height(len(districts)),
        right_margin=80,
        bottom_margin=80,
    )

    # Draw larger districts first so small districts are less likely to disappear.
    district_counts = {district: 0 for district in districts}
    for _x, _y, district in points:
        district_counts[district] += 1
    points.sort(key=lambda item: district_counts[item[2]], reverse=True)

    for x, y, district in points:
        draw_square(draw, x, y, point_radius, colors[district])

    draw_legend(image, colors, f"State FIPS {state_fips} Districts")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"state_{state_fips}_districts.png"
    image.save(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assignment_csv", nargs="?", type=Path, help="district assignment CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("district_maps"))
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--point-radius", type=int, default=1, help="0 draws one pixel per row")
    parser.add_argument("--seed", type=int, default=42, help="random color seed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.width < 200 or args.height < 200:
        raise ValueError("--width and --height must be at least 200")
    if args.point_radius < 0:
        raise ValueError("--point-radius must be 0 or greater")

    assignment_path = args.assignment_csv or prompt_path(
        "District assignment CSV",
        Path("Predictions_seed_grow_districts.csv"),
    )
    states = read_assignments(assignment_path)

    print(f"Loaded {sum(len(rows) for rows in states.values())} rows from {assignment_path}")
    print(f"Rendering {len(states)} state map(s) to {args.output_dir}")
    for state_fips, rows in sorted(states.items()):
        output_path = render_state(
            state_fips,
            rows,
            args.output_dir,
            args.width,
            args.height,
            args.point_radius,
            args.seed,
        )
        print(f"  {state_fips}: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, csv.Error) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2)
