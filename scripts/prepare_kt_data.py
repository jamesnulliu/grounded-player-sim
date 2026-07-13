"""Preprocess raw ASSISTments 2009 ("skill builder") data into the 6-column
TSV ``gps.data.kt_csv.load_kt_csv`` reads: ``user_id, item_id, timestamp,
correct, skill_id, ms_first_response``.

Mirrors the standard ``theophilee/learner-performance-prediction`` recipe for
"assistments09" (filter non-binary outcomes, drop untagged-skill rows,
dedupe multi-skill rows to one row per interaction, sort by ``order_id`` then
group by student preserving order) and additionally carries through
``ms_first_response`` (dropped by that recipe) so the *real* response-time
channel is available -- this is what makes the RQ5 when-not-what asymmetry
testable on real, non-synthetic timing (not just chess).

The raw file is not redistributed here; fetch it yourself, e.g. the USTC
mirror EduData points at:
  curl -O http://base.ustc.edu.cn/data/ASSISTment/2009_skill_builder_data_corrected.zip
  unzip 2009_skill_builder_data_corrected.zip

Usage:
  python scripts/prepare_kt_data.py skill_builder_data_corrected.csv out.tsv
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict


def prepare_assistments09(raw_path: str, out_path: str) -> None:
    seen_order_ids: set[str] = set()
    rows: list[tuple[int, str, str, str, str, str, str]] = []
    # (order_id_int, order_id_str, user_id, item_id, correct, skill_id, ms)

    with open(raw_path, encoding="ISO-8859-1", newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            order_id = r["order_id"]
            if order_id in seen_order_ids:
                continue  # keep first skill tag for multi-skill items
            correct = r["correct"]
            if correct not in ("0", "1"):
                continue  # drop continuous/partial-credit outcomes
            skill_id = r["skill_id"]
            if not skill_id:
                continue  # drop untagged-skill rows
            seen_order_ids.add(order_id)
            rows.append(
                (
                    int(order_id),
                    r["user_id"],
                    r["problem_id"],
                    correct,
                    skill_id,
                    r["ms_first_response"],
                )
            )

    rows.sort(key=lambda x: x[0])  # temporal proxy, per theophilee's recipe

    by_user: dict[str, list[tuple]] = defaultdict(list)
    for row in rows:
        by_user[row[1]].append(row)

    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(
            ["user_id", "item_id", "timestamp", "correct", "skill_id", "ms"]
        )
        n = 0
        for user_id, seq in by_user.items():
            # No real wall-clock timestamp in this dataset (per theophilee's
            # recipe); order_id already gives the temporal order used above.
            for _oid, _u, item_id, correct, skill_id, ms in seq:
                writer.writerow([user_id, item_id, 0, correct, skill_id, ms])
                n += 1

    print(
        f"{out_path}: {len(by_user)} students, {len(rows)} responses, "
        f"{len({r[4] for r in rows})} skills"
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(1)
    prepare_assistments09(sys.argv[1], sys.argv[2])
