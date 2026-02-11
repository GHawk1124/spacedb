#!/usr/bin/env python3

import argparse
import csv
import gzip
import json
from pathlib import Path


def num(value, cast=float):
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return cast(value)
    except (TypeError, ValueError):
        return None


def load_json(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def mid(a, b):
    if a is not None and b is not None:
        return (a + b) / 2.0
    return a if a is not None else b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cross_section_min", type=float)
    parser.add_argument("cross_section_max", type=float)
    parser.add_argument("altitude_min", type=float)
    parser.add_argument("altitude_max", type=float)
    args = parser.parse_args()
    ca_min, ca_max, alt_min, alt_max = (
        args.cross_section_min,
        args.cross_section_max,
        args.altitude_min,
        args.altitude_max,
    )
    if ca_min > ca_max:
        raise SystemExit("cross_section_min must be <= cross_section_max")
    if alt_min > alt_max:
        raise SystemExit("altitude_min must be <= altitude_max")

    root = Path(__file__).resolve().parent
    discos_path = root / "data/cache/discos_objects_by_satno.json.gz"
    satcat_path = root / "data/cache/spacetrack_satcat.json.gz"
    gp_path = root / "data/cache/spacetrack_gp.json.gz"
    output_path = root / "norads-ca-alt.csv"

    paths = [discos_path, satcat_path, gp_path]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"Required cache file(s) missing: {', '.join(missing)}")

    areas = {}
    for key, rec in load_json(discos_path).items():
        norad = num(rec.get("satno") if isinstance(rec, dict) else key, int)
        if norad is None or not isinstance(rec, dict):
            continue
        x_min, x_avg, x_max = (
            num(rec.get("xSectMin")),
            num(rec.get("xSectAvg")),
            num(rec.get("xSectMax")),
        )
        area = x_avg if x_avg is not None else mid(x_min, x_max)
        if area is not None:
            areas[norad] = area

    altitudes = {}
    for rows, apo, per in (
        (load_json(satcat_path), "APOGEE", "PERIGEE"),
        (load_json(gp_path), "APOAPSIS", "PERIAPSIS"),
    ):
        for rec in rows:
            norad = num(rec.get("NORAD_CAT_ID") if isinstance(rec, dict) else None, int)
            alt = mid(
                num(rec.get(apo) if isinstance(rec, dict) else None),
                num(rec.get(per) if isinstance(rec, dict) else None),
            )
            if norad is not None and alt is not None:
                altitudes[norad] = alt

    rows = [
        (norad, areas[norad], altitudes[norad])
        for norad in sorted(areas.keys() & altitudes.keys())
        if ca_min <= areas[norad] <= ca_max and alt_min <= altitudes[norad] <= alt_max
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(
            [["norad_cat_id", "cross_section_area_m2", "altitude_km"], *rows]
        )

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
