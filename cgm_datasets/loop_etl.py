#!/usr/bin/env python3
"""Convert raw Loop exports into the prepared CGM format used by this repo."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cgm_datasets.common import (
    AI_READI_GLUCOSE_MEAN_MGDL,
    AI_READI_GLUCOSE_STD_MGDL,
    DEFAULT_EVAL_WINDOW,
    DEFAULT_SAMPLING_MINUTES,
    DatasetMetadata,
    ensure_dir,
    maybe_convert_mmol_to_mgdl,
    normalize_glucose_mgdl,
    standardize_record,
    summarize_splits,
    train_val_test_split,
)

DATASET_NAME = "loop"
DEFAULT_OUT_DIR = os.path.join("data", "prepared", DATASET_NAME)
GRID_FREQ = f"{DEFAULT_SAMPLING_MINUTES}min"
MIN_SEQ_LEN = DEFAULT_EVAL_WINDOW[1]

PERSON_ID_COLUMNS = (
    "PtID",
    "ptid",
    "person_id",
    "participant_id",
    "patient_id",
    "subject_id",
)
RECORD_TYPE_COLUMNS = ("RecordType", "record_type", "type", "event_type")
UNIT_COLUMNS = ("Units", "unit", "glucose_units")
CGM_TIMESTAMP_COLUMNS = ("UTCDtTm", "DeviceDtTm", "timestamp", "datetime", "time")
CGM_VALUE_COLUMNS = ("CGMVal", "glucose", "sensor_glucose", "glucose_value")
MEAL_TIMESTAMP_COLUMNS = ("MealDtTm", "timestamp", "datetime", "time")
MEAL_VALUE_COLUMNS = (
    "CarbInput",
    "CarbGrams",
    "carbs",
    "meal_carbs",
    "carbohydrates",
    "food",
)
EXERCISE_START_COLUMNS = (
    "ExerciseDtTm",
    "StartDtTm",
    "start_time",
    "timestamp",
    "datetime",
    "time",
)
EXERCISE_END_COLUMNS = ("EndDtTm", "StopDtTm", "end_time", "end_datetime")
EXERCISE_DURATION_COLUMNS = (
    "DurationMin",
    "duration_min",
    "duration_minutes",
    "minutes",
    "duration",
)


def _normalize_col(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace("(", " ")
        .replace(")", " ")
        .replace("/", " ")
        .replace("-", "_")
        .replace("%", "pct")
        .replace("__", "_")
        .replace(" ", "_")
    )


def _find_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {_normalize_col(col): col for col in columns}
    for candidate in candidates:
        key = _normalize_col(candidate)
        if key in lowered:
            return lowered[key]
    for normalized, original in lowered.items():
        if any(_normalize_col(candidate) in normalized for candidate in candidates):
            return original
    return None


def _normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _normalize_person_token(value: Any) -> str | None:
    text = _normalize_text(value)
    if not text or text in {"nan", "none"}:
        return None
    try:
        return str(int(float(text)))
    except Exception:
        return text


def _coerce_datetime(series: pd.Series) -> pd.Series:
    out = pd.to_datetime(series, errors="coerce", utc=False)
    if out.notna().sum() == 0:
        raise ValueError("could not parse any timestamps")
    try:
        tz = out.dt.tz
    except AttributeError:
        tz = None
    if tz is not None:
        out = out.dt.tz_localize(None)
    return out


def _tabular_files(root_dir: str) -> list[str]:
    files: list[str] = []
    for root, _, names in os.walk(root_dir):
        if "__MACOSX" in root.split(os.sep):
            continue
        for name in names:
            if name.startswith(".") or name.startswith("._"):
                continue
            if name.lower().endswith((".csv", ".tsv", ".txt")):
                files.append(os.path.join(root, name))
    return sorted(files)


def _read_table(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        header = handle.readline()
    pipe_count = header.count("|")
    comma_count = header.count(",")
    tab_count = header.count("\t")
    if pipe_count > 0 and pipe_count >= max(comma_count, tab_count):
        return pd.read_csv(path, sep="|")
    if tab_count > 0 and tab_count > comma_count:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path, sep=None, engine="python")


def _row_type_mask(
    df: pd.DataFrame,
    *,
    path: str,
    row_tokens: tuple[str, ...],
    filename_tokens: tuple[str, ...],
) -> pd.Series | None:
    record_type_col = _find_column(list(df.columns), RECORD_TYPE_COLUMNS)
    lowered_path = os.path.basename(path).lower()
    if record_type_col is not None:
        normalized = df[record_type_col].map(_normalize_text)
        mask = normalized.map(lambda value: any(token in value for token in row_tokens))
        if mask.any():
            return mask
    if any(token in lowered_path for token in filename_tokens):
        return pd.Series(True, index=df.index)
    return None


def _convert_glucose_to_mgdl(
    values: pd.Series, units: pd.Series | None
) -> tuple[np.ndarray, bool]:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float32)
    if units is not None:
        unit_values = {
            _normalize_text(item).replace(" ", "")
            for item in units
            if _normalize_text(item)
        }
        has_mmol = any("mmol" in item for item in unit_values)
        has_mg = any("mg" in item for item in unit_values)
        if has_mmol and has_mg:
            raise ValueError(f"mixed glucose units in one file: {sorted(unit_values)}")
        if has_mmol:
            return (arr * 18.0).astype(np.float32), True
        if has_mg:
            return arr.astype(np.float32), False
    converted, was_converted = maybe_convert_mmol_to_mgdl(arr)
    return converted.astype(np.float32), bool(was_converted)


def _extract_cgm_rows(df: pd.DataFrame, path: str) -> pd.DataFrame | None:
    cols = list(df.columns)
    person_col = _find_column(cols, PERSON_ID_COLUMNS)
    time_col = _find_column(cols, CGM_TIMESTAMP_COLUMNS)
    value_col = _find_column(cols, CGM_VALUE_COLUMNS)
    if person_col is None or time_col is None or value_col is None:
        return None

    row_mask = _row_type_mask(
        df,
        path=path,
        row_tokens=("cgm", "glucose", "sensor"),
        filename_tokens=("cgm", "glucose", "dexcom", "medtronic"),
    )
    work = df.loc[row_mask].copy() if row_mask is not None else df.copy()
    if work.empty:
        return None

    work["__raw_person_id__"] = work[person_col].map(_normalize_person_token)
    try:
        work["__timestamp__"] = _coerce_datetime(work[time_col])
    except ValueError:
        return None
    units_col = _find_column(cols, UNIT_COLUMNS)
    glucose_mgdl, converted = _convert_glucose_to_mgdl(
        work[value_col],
        work[units_col] if units_col is not None else None,
    )
    work["__glucose_mgdl__"] = glucose_mgdl
    work["__converted_from_mmol_l__"] = int(converted)
    work = work.dropna(
        subset=["__raw_person_id__", "__timestamp__", "__glucose_mgdl__"]
    )
    if work.empty:
        return None
    return work[
        [
            "__raw_person_id__",
            "__timestamp__",
            "__glucose_mgdl__",
            "__converted_from_mmol_l__",
        ]
    ]


def _extract_meal_events(df: pd.DataFrame, path: str) -> list[dict[str, Any]]:
    cols = list(df.columns)
    person_col = _find_column(cols, PERSON_ID_COLUMNS)
    time_col = _find_column(cols, MEAL_TIMESTAMP_COLUMNS)
    value_col = _find_column(cols, MEAL_VALUE_COLUMNS)
    if person_col is None or time_col is None:
        return []

    row_mask = _row_type_mask(
        df,
        path=path,
        row_tokens=("meal", "carb", "food"),
        filename_tokens=("meal", "carb", "food"),
    )
    if row_mask is None and value_col is None:
        return []

    work = df.loc[row_mask].copy() if row_mask is not None else df.copy()
    if work.empty:
        return []
    work["__raw_person_id__"] = work[person_col].map(_normalize_person_token)
    try:
        work["__timestamp__"] = _coerce_datetime(work[time_col])
    except ValueError:
        return []
    if value_col is not None:
        numeric_value = pd.to_numeric(work[value_col], errors="coerce")
        has_positive_value = numeric_value.fillna(0).to_numpy() > 0
        if has_positive_value.any():
            work = work.loc[has_positive_value]
    work = work.dropna(subset=["__raw_person_id__", "__timestamp__"])
    if work.empty:
        return []
    return [
        {
            "raw_person_id": row["__raw_person_id__"],
            "start": row["__timestamp__"],
            "end": row["__timestamp__"],
        }
        for _, row in work.iterrows()
    ]


def _extract_exercise_events(df: pd.DataFrame, path: str) -> list[dict[str, Any]]:
    cols = list(df.columns)
    person_col = _find_column(cols, PERSON_ID_COLUMNS)
    start_col = _find_column(cols, EXERCISE_START_COLUMNS)
    if person_col is None or start_col is None:
        return []

    end_col = _find_column(cols, EXERCISE_END_COLUMNS)
    duration_col = _find_column(cols, EXERCISE_DURATION_COLUMNS)
    row_mask = _row_type_mask(
        df,
        path=path,
        row_tokens=("exercise", "activity", "workout", "run", "walk", "bike"),
        filename_tokens=("exercise", "activity", "workout"),
    )
    if row_mask is None and end_col is None and duration_col is None:
        return []

    work = df.loc[row_mask].copy() if row_mask is not None else df.copy()
    if work.empty:
        return []
    work["__raw_person_id__"] = work[person_col].map(_normalize_person_token)
    try:
        work["__start__"] = _coerce_datetime(work[start_col])
    except ValueError:
        return []
    if end_col is not None:
        try:
            work["__end__"] = _coerce_datetime(work[end_col])
        except ValueError:
            work["__end__"] = pd.NaT
    else:
        work["__end__"] = pd.NaT
    if duration_col is not None:
        duration_minutes = pd.to_numeric(work[duration_col], errors="coerce")
        fill_mask = (
            work["__end__"].isna() & duration_minutes.notna() & (duration_minutes > 0)
        )
        work.loc[fill_mask, "__end__"] = work.loc[
            fill_mask, "__start__"
        ] + pd.to_timedelta(duration_minutes.loc[fill_mask], unit="m")

    work = work.dropna(subset=["__raw_person_id__", "__start__"])
    if work.empty:
        return []

    events: list[dict[str, Any]] = []
    for _, row in work.iterrows():
        start = pd.Timestamp(row["__start__"])
        end = row["__end__"]
        if pd.isna(end):
            end_ts = start
        else:
            end_ts = pd.Timestamp(end)
            if end_ts < start:
                end_ts = start
        events.append(
            {"raw_person_id": row["__raw_person_id__"], "start": start, "end": end_ts}
        )
    return events


def _aligned_grid(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    start = index.min().floor("D")
    end = (
        index.max().floor("D")
        + pd.Timedelta(days=1)
        - pd.Timedelta(minutes=DEFAULT_SAMPLING_MINUTES)
    )
    return pd.date_range(start=start, end=end, freq=GRID_FREQ)


def _build_person_id_map(raw_ids: list[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    max_numeric = 0
    non_numeric: list[str] = []
    for raw_id in sorted(set(raw_ids)):
        try:
            numeric = int(raw_id)
        except Exception:
            non_numeric.append(raw_id)
            continue
        mapping[raw_id] = numeric
        max_numeric = max(max_numeric, numeric)
    next_id = max_numeric + 1 if max_numeric > 0 else 1
    for raw_id in non_numeric:
        mapping[raw_id] = next_id
        next_id += 1
    return mapping


def _rasterize_event_proxy(
    grid: pd.DatetimeIndex,
    meal_events: list[dict[str, Any]],
    exercise_events: list[dict[str, Any]],
) -> np.ndarray:
    proxy = np.zeros(len(grid), dtype=np.float32)

    def add_span(start: pd.Timestamp, end: pd.Timestamp) -> None:
        start_bin = pd.Timestamp(start).floor(GRID_FREQ)
        end_ts = pd.Timestamp(end)
        if end_ts <= start:
            end_bin = start_bin + pd.Timedelta(minutes=DEFAULT_SAMPLING_MINUTES)
        else:
            end_bin = end_ts.ceil(GRID_FREQ)
        lo = int(grid.searchsorted(start_bin, side="left"))
        hi = int(grid.searchsorted(end_bin, side="left"))
        if hi <= lo:
            hi = lo + 1
        lo = max(lo, 0)
        hi = min(hi, len(proxy))
        if lo < hi:
            proxy[lo:hi] += 1.0

    for event in meal_events:
        add_span(event["start"], event["end"])
    for event in exercise_events:
        add_span(event["start"], event["end"])
    return proxy.reshape(-1, 1)


def _missing_data_message(loop_raw_dir: str | None, reason: str) -> str:
    requested = loop_raw_dir or "<missing --loop_raw_dir>"
    return (
        f"Loop ETL could not run for --loop_raw_dir={requested}: {reason}\n"
        "The raw Loop export is not bundled with this repository. "
        "See cgm_datasets/loop_etl.md for the dry run, expected schema, and real command."
    )


def _validate_raw_dir(loop_raw_dir: str | None) -> list[str]:
    if not loop_raw_dir:
        raise FileNotFoundError("missing --loop_raw_dir")
    if not os.path.isdir(loop_raw_dir):
        raise FileNotFoundError("directory does not exist")
    files = _tabular_files(loop_raw_dir)
    if not files:
        raise FileNotFoundError(
            "directory is empty or contains no .csv/.tsv/.txt files"
        )
    return files


def _expected_schema_payload(out_dir: str, loop_raw_dir: str | None) -> dict[str, Any]:
    abs_out_dir = os.path.abspath(out_dir)
    return {
        "dataset": DATASET_NAME,
        "dry_run": True,
        "loop_raw_dir": os.path.abspath(loop_raw_dir) if loop_raw_dir else None,
        "expected_schema": {
            "cgm": {
                "participant_id_columns": list(PERSON_ID_COLUMNS),
                "timestamp_columns": list(CGM_TIMESTAMP_COLUMNS),
                "value_columns": list(CGM_VALUE_COLUMNS),
                "optional_units_columns": list(UNIT_COLUMNS),
                "optional_record_type_columns": list(RECORD_TYPE_COLUMNS),
            },
            "meal_events": {
                "participant_id_columns": list(PERSON_ID_COLUMNS),
                "timestamp_columns": list(MEAL_TIMESTAMP_COLUMNS),
                "optional_value_or_descriptor_columns": list(MEAL_VALUE_COLUMNS),
                "row_type_tokens": ["meal", "carb", "food"],
            },
            "exercise_events": {
                "participant_id_columns": list(PERSON_ID_COLUMNS),
                "start_timestamp_columns": list(EXERCISE_START_COLUMNS),
                "optional_end_timestamp_columns": list(EXERCISE_END_COLUMNS),
                "optional_duration_columns": list(EXERCISE_DURATION_COLUMNS),
                "row_type_tokens": [
                    "exercise",
                    "activity",
                    "workout",
                    "run",
                    "walk",
                    "bike",
                ],
            },
        },
        "processing_contract": {
            "sampling_minutes": DEFAULT_SAMPLING_MINUTES,
            "grid_alignment": "midnight-aligned 5-minute grid",
            "record_keys": [
                "irg_ts",
                "irg_ts_mask",
                "person_id",
                "label",
                "recommended_split",
                "seq_len",
                "ts_tt",
                "sampling_minutes",
            ],
            "metadata": {
                "name": DATASET_NAME,
                "glucose_mean_mgdl": AI_READI_GLUCOSE_MEAN_MGDL,
                "glucose_std_mgdl": AI_READI_GLUCOSE_STD_MGDL,
                "eval_window": list(DEFAULT_EVAL_WINDOW),
            },
            "activity_pkl_note": "loop_activity_test.pkl uses mm_steps as the meal+exercise trigger channel and fills mm_hr/mm_cal with zeros for benchmark compatibility.",
        },
        "outputs": {
            "prepared_dir": abs_out_dir,
            "split_pickles": [
                os.path.join(abs_out_dir, "loop_train.pkl"),
                os.path.join(abs_out_dir, "loop_val.pkl"),
                os.path.join(abs_out_dir, "loop_test.pkl"),
            ],
            "activity_test_pkl": os.path.join(abs_out_dir, "loop_activity_test.pkl"),
            "metadata_json": os.path.join(abs_out_dir, "metadata.json"),
        },
    }


def convert_loop_dataset(loop_raw_dir: str, out_dir: str) -> dict[str, Any]:
    files = _validate_raw_dir(loop_raw_dir)
    cgm_frames: list[pd.DataFrame] = []
    meal_events: list[dict[str, Any]] = []
    exercise_events: list[dict[str, Any]] = []
    detection_summary: dict[str, dict[str, int]] = {}

    for path in files:
        try:
            df = _read_table(path)
        except Exception as exc:
            raise ValueError(f"failed to read {path}: {exc}") from exc
        if df.empty:
            continue
        cgm_rows = _extract_cgm_rows(df, path)
        if cgm_rows is not None and not cgm_rows.empty:
            cgm_frames.append(cgm_rows)
            detection_summary[path] = detection_summary.get(path, {})
            detection_summary[path]["cgm_rows"] = int(len(cgm_rows))
        meal_rows = _extract_meal_events(df, path)
        if meal_rows:
            meal_events.extend(meal_rows)
            detection_summary[path] = detection_summary.get(path, {})
            detection_summary[path]["meal_events"] = int(len(meal_rows))
        exercise_rows = _extract_exercise_events(df, path)
        if exercise_rows:
            exercise_events.extend(exercise_rows)
            detection_summary[path] = detection_summary.get(path, {})
            detection_summary[path]["exercise_events"] = int(len(exercise_rows))

    if not cgm_frames:
        raise ValueError(
            "no Loop CGM table matched the documented schema assumptions; "
            "VERIFY the raw export against the module docstring and cgm_datasets/loop_etl.md"
        )
    if not meal_events and not exercise_events:
        raise ValueError(
            "no Loop meal/exercise tables matched the documented schema assumptions; "
            "VERIFY the raw export against the module docstring and cgm_datasets/loop_etl.md"
        )

    cgm_all = pd.concat(cgm_frames, ignore_index=True)
    if cgm_all.empty:
        raise ValueError(
            "matched Loop CGM tables, but none contained usable timestamp/value rows"
        )

    raw_person_ids = cgm_all["__raw_person_id__"].dropna().astype(str).tolist()
    if not raw_person_ids:
        raise ValueError(
            "matched Loop CGM rows, but none had usable participant identifiers"
        )
    person_id_map = _build_person_id_map(raw_person_ids)

    meals_by_person: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in meal_events:
        meals_by_person[str(event["raw_person_id"])].append(event)
    exercise_by_person: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in exercise_events:
        exercise_by_person[str(event["raw_person_id"])].append(event)

    records: list[dict[str, Any]] = []
    activity_lookup: dict[int, np.ndarray] = {}
    skipped_short = 0
    converted_participants = 0

    for raw_person_id, person_rows in cgm_all.groupby("__raw_person_id__"):
        ordered = person_rows.sort_values("__timestamp__").copy()
        cgm_series = (
            ordered.set_index("__timestamp__")["__glucose_mgdl__"]
            .sort_index()
            .resample(GRID_FREQ)
            .mean()
        )
        if cgm_series.empty:
            continue
        grid = _aligned_grid(cgm_series.index)
        cgm_series = cgm_series.reindex(grid)
        if len(grid) < MIN_SEQ_LEN:
            skipped_short += 1
            continue

        person_id = person_id_map[str(raw_person_id)]
        observed_mask = cgm_series.notna().to_numpy(dtype=np.float32)
        glucose_z = normalize_glucose_mgdl(
            np.nan_to_num(cgm_series.to_numpy(dtype=np.float32), nan=0.0)
        )
        if int(ordered["__converted_from_mmol_l__"].max()) > 0:
            converted_participants += 1

        record = standardize_record(
            glucose_z,
            observed_mask,
            person_id=person_id,
            split="train",
            label=1,
            dataset_name=DATASET_NAME,
            extras={
                "source_person_id": str(raw_person_id),
                "raw_start_time": str(pd.Timestamp(ordered["__timestamp__"].iloc[0])),
                "raw_end_time": str(pd.Timestamp(ordered["__timestamp__"].iloc[-1])),
                "converted_from_mmol_l": int(
                    ordered["__converted_from_mmol_l__"].max()
                ),
            },
        )
        records.append(record)
        activity_lookup[person_id] = _rasterize_event_proxy(
            grid,
            meals_by_person.get(str(raw_person_id), []),
            exercise_by_person.get(str(raw_person_id), []),
        ).astype(np.float32)

    if not records:
        raise ValueError(
            "no Loop participants produced usable CGM records after parsing and length validation; "
            f"need at least {MIN_SEQ_LEN} aligned 5-minute samples per participant for the benchmark"
        )
    if all(float(activity.sum()) <= 0.0 for activity in activity_lookup.values()):
        raise ValueError(
            "Loop activity tables were detected, but no meal/exercise events aligned to the parsed CGM participants; "
            "VERIFY participant ids and timestamp columns"
        )

    splits = train_val_test_split(records, seed=42)
    test_activity: list[dict[str, Any]] = []
    for record in splits["test"]:
        person_id = int(record["person_id"])
        steps = activity_lookup[person_id].astype(np.float32)
        zeros = np.zeros_like(steps, dtype=np.float32)
        test_activity.append(
            {
                "person_id": person_id,
                "mm_steps": steps,
                "mm_hr": zeros.copy(),
                "mm_cal": zeros.copy(),
            }
        )

    abs_out_dir = os.path.abspath(out_dir)
    ensure_dir(abs_out_dir)
    for split_name in ("train", "val", "test"):
        with open(os.path.join(abs_out_dir, f"loop_{split_name}.pkl"), "wb") as handle:
            pickle.dump(splits[split_name], handle)
    with open(os.path.join(abs_out_dir, "loop_activity_test.pkl"), "wb") as handle:
        pickle.dump(test_activity, handle)

    metadata = DatasetMetadata(
        name=DATASET_NAME,
        display_name="Loop Observational Study",
        description=(
            "Loop CGM cohort resampled to a 5-minute midnight-aligned grid for the Toye benchmark. "
            "Activity sidecar uses meal+exercise events as the MAR trigger channel."
        ),
        source_dir=abs_out_dir,
        raw_source=os.path.abspath(loop_raw_dir),
        sampling_minutes=DEFAULT_SAMPLING_MINUTES,
        glucose_mean_mgdl=AI_READI_GLUCOSE_MEAN_MGDL,
        glucose_std_mgdl=AI_READI_GLUCOSE_STD_MGDL,
        supported_modalities=["cgm"],
        cohort_counts={"t1d": len({int(record["person_id"]) for record in records})},
        notes={
            "split_summary": summarize_splits(splits),
            "activity_test_pkl": "loop_activity_test.pkl",
            "activity_definition": "mm_steps is the 5-minute meal+exercise event count; mm_hr and mm_cal are zero placeholders for benchmark compatibility",
            "schema_assumptions": "VERIFY against real Loop export; see cgm_datasets/loop_etl.py module docstring",
            "detected_tables": detection_summary,
            "skipped_short_sequences": skipped_short,
            "min_required_seq_len": MIN_SEQ_LEN,
            "participants_converted_from_mmol_l": converted_participants,
        },
    )
    with open(
        os.path.join(abs_out_dir, "metadata.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata.to_dict(), handle, indent=2, sort_keys=True)

    return {
        "dataset": DATASET_NAME,
        "out_dir": abs_out_dir,
        "n_participants": len(records),
        "split_summary": summarize_splits(splits),
        "activity_test_records": len(test_activity),
        "skipped_short_sequences": skipped_short,
        "participants_converted_from_mmol_l": converted_participants,
        "files_written": [
            os.path.join(abs_out_dir, "loop_train.pkl"),
            os.path.join(abs_out_dir, "loop_val.pkl"),
            os.path.join(abs_out_dir, "loop_test.pkl"),
            os.path.join(abs_out_dir, "loop_activity_test.pkl"),
            os.path.join(abs_out_dir, "metadata.json"),
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert raw Loop exports into the prepared 5-minute CGM format used by this repo."
    )
    parser.add_argument(
        "--loop_raw_dir",
        default=None,
        help="Directory containing raw Loop tabular exports (.csv/.tsv/.txt). Required unless --dry_run is set.",
    )
    parser.add_argument(
        "--out_dir",
        default=DEFAULT_OUT_DIR,
        help=f"Prepared output directory. Defaults to {DEFAULT_OUT_DIR}.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the expected raw schema and output paths without reading any Loop data.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dry_run:
        print(
            json.dumps(
                _expected_schema_payload(args.out_dir, args.loop_raw_dir), indent=2
            )
        )
        return 0

    try:
        summary = convert_loop_dataset(args.loop_raw_dir, args.out_dir)
    except FileNotFoundError as exc:
        print(_missing_data_message(args.loop_raw_dir, str(exc)), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(
            f"Loop ETL failed: {exc}\n"
            "See cgm_datasets/loop_etl.md for the expected schema, dry run, and real command.",
            file=sys.stderr,
        )
        return 1

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
