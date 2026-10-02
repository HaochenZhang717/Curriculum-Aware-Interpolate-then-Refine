from __future__ import annotations

import abc
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from cgm_datasets.common import (
    AI_READI_GLUCOSE_MEAN_MGDL,
    AI_READI_GLUCOSE_STD_MGDL,
    DEFAULT_DAY_LEN,
    DEFAULT_EVAL_WINDOW,
    DatasetMetadata,
    load_prepared_dataset,
    maybe_convert_mmol_to_mgdl,
    normalize_glucose_mgdl,
    prepared_dir_has_splits,
    save_prepared_dataset,
    standardize_record,
    summarize_splits,
    train_val_test_split,
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
        if any(candidate in normalized for candidate in candidates):
            return original
    return None


def _coerce_datetime(series: pd.Series) -> pd.Series:
    out = pd.to_datetime(series, errors="coerce", utc=False)
    if out.notna().sum() == 0:
        raise ValueError("could not parse any timestamps")
    return out


def _resample_numeric(
    df: pd.DataFrame,
    timestamp_col: str,
    value_col: str,
    *,
    agg: str,
) -> pd.Series:
    series = (
        df[[timestamp_col, value_col]]
        .dropna(subset=[timestamp_col])
        .assign(**{timestamp_col: _coerce_datetime(df[timestamp_col])})
        .dropna(subset=[timestamp_col])
        .set_index(timestamp_col)[value_col]
        .sort_index()
    )
    series = pd.to_numeric(series, errors="coerce")
    if agg == "sum":
        return series.resample("5min").sum(min_count=1)
    if agg == "max":
        return series.resample("5min").max()
    return series.resample("5min").mean()


def _aligned_grid(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    start = index.min().floor("D")
    end = index.max().ceil("D") - pd.Timedelta(minutes=5)
    return pd.date_range(start=start, end=end, freq="5min")


def _event_array(
    df: pd.DataFrame,
    timestamp_col: str,
    columns: tuple[str, ...],
    *,
    agg: str,
    grid: pd.DatetimeIndex,
) -> np.ndarray | None:
    col = _find_column(list(df.columns), columns)
    if col is None:
        return None
    series = _resample_numeric(df, timestamp_col, col, agg=agg).reindex(
        grid, fill_value=0.0
    )
    return np.nan_to_num(series.to_numpy(dtype=np.float32), nan=0.0).reshape(-1, 1)


def _binary_event_array(
    df: pd.DataFrame,
    timestamp_col: str,
    columns: tuple[str, ...],
    *,
    grid: pd.DatetimeIndex,
) -> np.ndarray | None:
    col = _find_column(list(df.columns), columns)
    if col is None:
        return None
    tmp = df[[timestamp_col, col]].copy()
    tmp[timestamp_col] = _coerce_datetime(tmp[timestamp_col])
    tmp = tmp.dropna(subset=[timestamp_col])
    if tmp.empty:
        return None

    def to_flag(value: Any) -> float:
        if pd.isna(value):
            return 0.0
        text = str(value).strip().lower()
        if not text:
            return 0.0
        if text in {
            "nan",
            "none",
            "未记录",
            "data not available",
            "not available",
            "/",
        }:
            return 0.0
        return 1.0

    tmp["__flag__"] = tmp[col].map(to_flag).astype(np.float32)
    series = (
        tmp.set_index(timestamp_col)["__flag__"]
        .sort_index()
        .resample("5min")
        .max()
        .reindex(grid, fill_value=0.0)
    )
    return np.nan_to_num(series.to_numpy(dtype=np.float32), nan=0.0).reshape(-1, 1)


def _tabular_files(source: str) -> list[str]:
    if os.path.isfile(source):
        return [source]
    files: list[str] = []
    for root, _, names in os.walk(source):
        if "__MACOSX" in root.split(os.sep):
            continue
        for name in names:
            if name.startswith("._") or name.startswith("."):
                continue
            lower = name.lower()
            if lower.endswith((".csv", ".tsv", ".txt", ".xlsx", ".xls")):
                files.append(os.path.join(root, name))
    return sorted(files)


def _read_table(path: str) -> pd.DataFrame:
    lower = path.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    if lower.endswith(".tsv") or lower.endswith(".txt"):
        return pd.read_csv(path, sep=None, engine="python")
    if lower.endswith(".xlsx"):
        return pd.read_excel(path, engine="openpyxl")
    if lower.endswith(".xls"):
        return pd.read_excel(path, engine="xlrd")
    raise ValueError(f"unsupported tabular file: {path}")


@dataclass
class AdapterResult:
    train: list[dict[str, Any]]
    val: list[dict[str, Any]]
    test: list[dict[str, Any]]
    metadata: DatasetMetadata


class BaseDatasetAdapter(abc.ABC):
    name: str
    aliases: tuple[str, ...]

    def __init__(self, root_dir: str):
        self.root_dir = root_dir

    @property
    def default_output_dir(self) -> str:
        return os.path.join(self.root_dir, "data", "prepared", self.name)

    @abc.abstractmethod
    def default_prepared_dirs(self) -> list[str]:
        raise NotImplementedError

    def load(self, prepared_dir: str | None = None) -> AdapterResult:
        dataset_dir = prepared_dir or self._resolve_existing_prepared_dir()
        train, val, test, metadata = load_prepared_dataset(dataset_dir, self.name)
        if metadata.source_dir is None:
            metadata.source_dir = dataset_dir
        return AdapterResult(train=train, val=val, test=test, metadata=metadata)

    def prepare(
        self, raw_source: str, output_dir: str | None = None, force: bool = False
    ) -> str:
        output_dir = output_dir or self.default_output_dir
        if os.path.exists(os.path.join(output_dir, "metadata.json")) and not force:
            raise FileExistsError(
                f"{output_dir} already exists; pass force=True to overwrite"
            )
        result = self._prepare_raw(raw_source)
        result.metadata.source_dir = output_dir
        save_prepared_dataset(
            output_dir,
            self.name,
            {"train": result.train, "val": result.val, "test": result.test},
            result.metadata,
        )
        return output_dir

    @abc.abstractmethod
    def _prepare_raw(self, raw_source: str) -> AdapterResult:
        raise NotImplementedError

    def _resolve_existing_prepared_dir(self) -> str:
        for path in self.default_prepared_dirs():
            if prepared_dir_has_splits(path, self.name):
                return path
        raise FileNotFoundError(
            f"could not locate prepared dataset for '{self.name}'. "
            f"Run `python -m cgm_datasets.prepare prepare --dataset {self.name} --source <raw_dir>` first."
        )


class AIReadiAdapter(BaseDatasetAdapter):
    name = "aireadi"
    aliases = ("ai-readi", "aireadi_cgm", "aireadi_cgm_full")

    def default_prepared_dirs(self) -> list[str]:
        return [
            os.path.join(self.root_dir, "data", "aireadi_cgm_full"),
            self.default_output_dir,
        ]

    def load(self, prepared_dir: str | None = None) -> AdapterResult:
        result = super().load(prepared_dir=prepared_dir)
        if result.metadata.display_name == result.metadata.name:
            result.metadata.display_name = "AI-READI CGM"
        if not result.metadata.supported_modalities:
            result.metadata.supported_modalities = [
                "cgm",
                "ecg_feats",
                "wearable_feats",
                "clinical_feats",
                "cfp_feats",
            ]
        return result

    def _prepare_raw(self, raw_source: str) -> AdapterResult:
        train, val, test, _ = load_prepared_dataset(raw_source, self.name)
        metadata = DatasetMetadata(
            name=self.name,
            display_name="AI-READI CGM",
            description="Legacy prepared AI-READI CGM split.",
            source_dir=raw_source,
            raw_source=raw_source,
            supported_modalities=[
                "cgm",
                "ecg_feats",
                "wearable_feats",
                "clinical_feats",
                "cfp_feats",
            ],
        )
        return AdapterResult(train=train, val=val, test=test, metadata=metadata)


class ShanghaiAdapter(BaseDatasetAdapter):
    name = "shanghai"
    aliases = ("shanghai_t1dm", "shanghai_t2dm", "shanghai_diabetes")

    def default_prepared_dirs(self) -> list[str]:
        return [self.default_output_dir]

    def _prepare_raw(self, raw_source: str) -> AdapterResult:
        clinical_lookup = self._load_summary_lookup(raw_source)
        records: list[dict[str, Any]] = []
        for path in _tabular_files(raw_source):
            lower = path.lower()
            if "summary" in lower:
                continue
            df = _read_table(path)
            if df.empty:
                continue
            timestamp_col = _find_column(
                list(df.columns), ("timestamp", "datetime", "time", "date")
            )
            glucose_col = _find_column(
                list(df.columns),
                ("cgm", "glucose", "sensor_glucose", "sg", "glucose_value", "cgm_mgdl"),
            )
            if timestamp_col is None or glucose_col is None:
                continue
            timestamp = _coerce_datetime(df[timestamp_col])
            glucose_mgdl, converted = maybe_convert_mmol_to_mgdl(
                pd.to_numeric(df[glucose_col], errors="coerce")
            )
            temp = df.copy()
            temp["__timestamp__"] = timestamp
            temp["__glucose_mgdl__"] = glucose_mgdl
            temp = temp.dropna(subset=["__timestamp__"])
            if temp.empty:
                continue
            cgm = (
                temp.set_index("__timestamp__")["__glucose_mgdl__"]
                .sort_index()
                .resample("5min")
                .mean()
            )
            grid = _aligned_grid(cgm.index)
            cgm = cgm.reindex(grid)
            mask = cgm.notna().to_numpy(dtype=np.float32)
            glucose_z = normalize_glucose_mgdl(
                np.nan_to_num(cgm.to_numpy(dtype=np.float32), nan=0.0)
            )

            record_id = os.path.splitext(os.path.basename(path))[0]
            person_id = record_id.split("_")[0]
            label = 1 if "t1" in lower else 2 if "t2" in lower else 0
            cohort = "t1d" if label == 1 else "t2d" if label == 2 else "unknown"
            meal_series = _binary_event_array(
                temp, "__timestamp__", ("dietary intake", "饮食", "进食量"), grid=grid
            )
            bolus_sc = _event_array(
                temp, "__timestamp__", ("insulin dose - s.c.",), agg="sum", grid=grid
            )
            bolus_csii = _event_array(
                temp,
                "__timestamp__",
                ("csii - bolus insulin", "csii bolus"),
                agg="sum",
                grid=grid,
            )
            basal_csii = _event_array(
                temp,
                "__timestamp__",
                ("csii - basal insulin", "胰岛素泵基础量"),
                agg="mean",
                grid=grid,
            )
            clinical = (
                clinical_lookup.get(record_id) or clinical_lookup.get(person_id) or {}
            )

            extras = {
                "cohort": cohort,
                "source_file": path,
                "record_id": record_id,
                "converted_from_mmol_l": int(converted),
            }
            if meal_series is not None:
                extras["meal_series"] = meal_series.astype(np.float32)
            if bolus_sc is not None:
                extras["sc_insulin_series"] = bolus_sc.astype(np.float32)
            if bolus_csii is not None:
                extras["bolus_insulin_series"] = bolus_csii.astype(np.float32)
            if basal_csii is not None:
                extras["basal_insulin_series"] = basal_csii.astype(np.float32)
            if clinical:
                extras["clinical_summary"] = clinical
            records.append(
                standardize_record(
                    glucose_z,
                    mask,
                    person_id=person_id,
                    split="train",
                    label=label,
                    dataset_name=self.name,
                    extras=extras,
                )
            )

        if not records:
            raise ValueError(f"no Shanghai CGM records found under {raw_source}")

        splits = train_val_test_split(records, seed=42)
        metadata = DatasetMetadata(
            name=self.name,
            display_name="Shanghai T1DM/T2DM",
            description="Open Shanghai diabetes cohort adapted to the AI-READI-compatible CGM schema.",
            raw_source=raw_source,
            source_dir=self.default_output_dir,
            supported_modalities=[
                "cgm",
                "meal_series",
                "sc_insulin_series",
                "bolus_insulin_series",
                "basal_insulin_series",
            ],
            cohort_counts={
                "t1d": sum(int(record.get("label", 0) == 1) for record in records),
                "t2d": sum(int(record.get("label", 0) == 2) for record in records),
            },
            notes={
                "split_summary": summarize_splits(splits),
                "meal_series_definition": "binary 5-minute meal-event flag derived from Dietary intake / 饮食 annotations",
            },
        )
        return AdapterResult(
            train=splits["train"],
            val=splits["val"],
            test=splits["test"],
            metadata=metadata,
        )

    def _load_summary_lookup(self, raw_source: str) -> dict[str, dict[str, Any]]:
        lookup: dict[str, dict[str, Any]] = {}
        for path in _tabular_files(raw_source):
            lower = os.path.basename(path).lower()
            if "summary" not in lower:
                continue
            try:
                df = _read_table(path)
            except Exception:
                continue
            if df.empty:
                continue
            patient_col = _find_column(
                list(df.columns), ("patient number", "patient_number")
            )
            if patient_col is None:
                continue
            for _, row in df.iterrows():
                patient = str(row.get(patient_col, "")).strip()
                if not patient:
                    continue
                flat = {}
                for col in df.columns:
                    val = row[col]
                    if isinstance(val, (np.generic,)):
                        val = val.item()
                    if pd.isna(val):
                        continue
                    flat[_normalize_col(str(col))] = val
                lookup[patient] = flat
                lookup[patient.split("_")[0]] = flat
        return lookup


class OhioT1DMAdapter(BaseDatasetAdapter):
    name = "ohio"
    aliases = ("ohiot1dm", "ohio_t1dm", "ohio-t1dm")

    def default_prepared_dirs(self) -> list[str]:
        return [self.default_output_dir]

    def _prepare_raw(self, raw_source: str) -> AdapterResult:
        xml_files = []
        for root, _, names in os.walk(raw_source):
            for fname in names:
                if fname.lower().endswith(".xml"):
                    xml_files.append(os.path.join(root, fname))
        if not xml_files:
            raise ValueError(f"no OhioT1DM .xml files found under {raw_source}")

        # one patient -> possibly multiple files (training + testing); merge by patient id
        by_patient: dict[str, list[tuple[pd.Timestamp, float]]] = {}
        for path in sorted(xml_files):
            tree = ET.parse(path)
            root_el = tree.getroot()
            pid = str(root_el.attrib.get("id", os.path.basename(path).split("-")[0]))
            samples = by_patient.setdefault(pid, [])
            for gl in root_el.iter("glucose_level"):
                for ev in gl.iter("event"):
                    ts = pd.to_datetime(
                        ev.attrib["ts"], format="%d-%m-%Y %H:%M:%S", errors="coerce"
                    )
                    val = pd.to_numeric(ev.attrib.get("value"), errors="coerce")
                    if pd.notna(ts) and pd.notna(val):
                        samples.append((ts, float(val)))

        records: list[dict[str, Any]] = []
        for pid, samples in by_patient.items():
            if not samples:
                continue
            ser = pd.Series({ts: v for ts, v in samples}).sort_index()
            glucose_mgdl, converted = maybe_convert_mmol_to_mgdl(
                ser.to_numpy(dtype=np.float32)
            )
            ser = pd.Series(glucose_mgdl, index=ser.index)
            cgm = ser.resample("5min").mean()
            grid = _aligned_grid(cgm.index)
            cgm = cgm.reindex(grid)
            mask = cgm.notna().to_numpy(dtype=np.float32)
            glucose_z = normalize_glucose_mgdl(
                np.nan_to_num(cgm.to_numpy(dtype=np.float32), nan=0.0)
            )
            records.append(
                standardize_record(
                    glucose_z,
                    mask,
                    person_id=pid,
                    split="train",
                    label=1,
                    dataset_name=self.name,
                    extras={
                        "cohort": "t1d",
                        "record_id": pid,
                        "converted_from_mmol_l": int(converted),
                    },
                )
            )

        if not records:
            raise ValueError(f"no usable OhioT1DM CGM records under {raw_source}")
        splits = train_val_test_split(records, seed=42)
        metadata = DatasetMetadata(
            name=self.name,
            display_name="OhioT1DM",
            description="OhioT1DM CGM cohort adapted to the AI-READI-compatible schema.",
            raw_source=raw_source,
            source_dir=self.default_output_dir,
            supported_modalities=["cgm"],
            cohort_counts={"t1d": len({str(r["person_id"]) for r in records})},
            notes={"split_summary": summarize_splits(splits)},
        )
        return AdapterResult(
            train=splits["train"],
            val=splits["val"],
            test=splits["test"],
            metadata=metadata,
        )


class HupaUcmAdapter(BaseDatasetAdapter):
    name = "hupa_ucm"
    aliases = ("hupa", "hupa-ucm", "hupa_ucm_t1d")

    def default_prepared_dirs(self) -> list[str]:
        return [self.default_output_dir]

    def _resolve_source_dirs(self, raw_source: str) -> tuple[str, str | None]:
        source = os.path.abspath(raw_source)
        if not os.path.exists(source):
            raise FileNotFoundError(source)

        if os.path.isfile(source):
            preprocessed_dir = os.path.dirname(source)
            parent = os.path.dirname(preprocessed_dir)
            raw_data_dir = os.path.join(parent, "Raw_Data")
            return preprocessed_dir, (
                raw_data_dir if os.path.isdir(raw_data_dir) else None
            )

        base = os.path.basename(source).lower()
        if base == "preprocessed":
            preprocessed_dir = source
            raw_data_dir = os.path.join(os.path.dirname(source), "Raw_Data")
        elif base == "raw_data":
            preprocessed_dir = os.path.join(os.path.dirname(source), "Preprocessed")
            raw_data_dir = source
        else:
            nested_preprocessed = os.path.join(source, "Preprocessed")
            preprocessed_dir = (
                nested_preprocessed if os.path.isdir(nested_preprocessed) else source
            )
            nested_raw = os.path.join(source, "Raw_Data")
            raw_data_dir = nested_raw if os.path.isdir(nested_raw) else None

        if not os.path.isdir(preprocessed_dir):
            raise FileNotFoundError(
                f"expected a HUPA-UCM Preprocessed directory under {raw_source}; "
                "point --source at the dataset root or its Preprocessed subdirectory"
            )
        if raw_data_dir is not None and not os.path.isdir(raw_data_dir):
            raw_data_dir = None
        return preprocessed_dir, raw_data_dir

    def _iter_preprocessed_files(self, preprocessed_dir: str) -> list[str]:
        files = []
        for name in sorted(os.listdir(preprocessed_dir)):
            if name.startswith("."):
                continue
            if not name.lower().endswith(".csv"):
                continue
            files.append(os.path.join(preprocessed_dir, name))
        return files

    def _read_preprocessed_csv(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path, sep=";")

    def _load_sleep_intervals(
        self, raw_data_dir: str | None
    ) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
        lookup: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
        if raw_data_dir is None:
            return lookup

        for person_id in sorted(os.listdir(raw_data_dir)):
            fitbit_dir = os.path.join(raw_data_dir, person_id, "fitbit")
            if not os.path.isdir(fitbit_dir):
                continue
            for name in sorted(os.listdir(fitbit_dir)):
                lower = name.lower()
                if (
                    "sleep" not in lower
                    or "summary" not in lower
                    or not lower.endswith(".csv")
                ):
                    continue
                path = os.path.join(fitbit_dir, name)
                try:
                    df = pd.read_csv(path)
                except Exception:
                    continue
                if df.empty:
                    continue
                start_col = _find_column(
                    list(df.columns), ("start time", "start_time", "start")
                )
                end_col = _find_column(
                    list(df.columns), ("end time", "end_time", "end")
                )
                if start_col is None or end_col is None:
                    continue
                try:
                    starts = _coerce_datetime(df[start_col])
                    ends = _coerce_datetime(df[end_col])
                except Exception:
                    continue
                for start, end in zip(starts, ends):
                    if pd.isna(start) or pd.isna(end):
                        continue
                    start_ts = pd.Timestamp(start)
                    end_ts = pd.Timestamp(end)
                    if end_ts <= start_ts:
                        continue
                    lookup.setdefault(person_id, []).append((start_ts, end_ts))
        return lookup

    def _sleep_array(
        self, intervals: list[tuple[pd.Timestamp, pd.Timestamp]], grid: pd.DatetimeIndex
    ) -> np.ndarray | None:
        if not intervals:
            return None
        sleep = np.zeros(len(grid), dtype=np.float32)
        for start_ts, end_ts in intervals:
            lo = int(grid.searchsorted(start_ts.floor("5min"), side="left"))
            hi = int(grid.searchsorted(end_ts.ceil("5min"), side="left"))
            if hi > lo:
                sleep[lo:hi] = 1.0
        return sleep.reshape(-1, 1)

    def _prepare_raw(self, raw_source: str) -> AdapterResult:
        preprocessed_dir, raw_data_dir = self._resolve_source_dirs(raw_source)
        sleep_lookup = self._load_sleep_intervals(raw_data_dir)
        records: list[dict[str, Any]] = []
        for path in self._iter_preprocessed_files(preprocessed_dir):
            df = self._read_preprocessed_csv(path)
            if df.empty:
                continue
            timestamp_col = _find_column(
                list(df.columns), ("time", "timestamp", "datetime", "date")
            )
            glucose_col = _find_column(
                list(df.columns),
                ("cgm", "glucose", "sensor_glucose", "glucose_value", "blood_glucose"),
            )
            if timestamp_col is None or glucose_col is None:
                continue

            df = df.copy()
            df[timestamp_col] = _coerce_datetime(df[timestamp_col])
            df = df.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
            df[glucose_col] = pd.to_numeric(df[glucose_col], errors="coerce")
            person_id = os.path.splitext(os.path.basename(path))[0]

            glucose_mgdl, converted = maybe_convert_mmol_to_mgdl(
                df[glucose_col].to_numpy(dtype=np.float32)
            )
            df = df.assign(__glucose_mgdl__=glucose_mgdl)
            cgm = _resample_numeric(df, timestamp_col, "__glucose_mgdl__", agg="mean")
            if cgm.empty:
                continue
            grid = _aligned_grid(cgm.index)
            cgm = cgm.reindex(grid)
            mask = cgm.notna().to_numpy(dtype=np.float32)
            glucose_z = normalize_glucose_mgdl(
                np.nan_to_num(cgm.to_numpy(dtype=np.float32), nan=0.0)
            )

            extras = {
                "cohort": "t1d",
                "source_file": path,
                "converted_from_mmol_l": int(converted),
                "raw_start_time": str(pd.Timestamp(df[timestamp_col].iloc[0])),
                "raw_end_time": str(pd.Timestamp(df[timestamp_col].iloc[-1])),
            }
            meals = _event_array(
                df,
                timestamp_col,
                ("carb_input", "carbs", "meal_carbs", "carbohydrates", "cho"),
                agg="sum",
                grid=grid,
            )
            bolus = _event_array(
                df,
                timestamp_col,
                ("bolus_volume_delivered", "bolus", "bolus_insulin", "insulin_bolus"),
                agg="sum",
                grid=grid,
            )
            basal = _event_array(
                df,
                timestamp_col,
                ("basal_rate", "basal", "basal_insulin"),
                agg="mean",
                grid=grid,
            )
            steps = _event_array(
                df, timestamp_col, ("steps", "step_count"), agg="sum", grid=grid
            )
            calories = _event_array(
                df, timestamp_col, ("calories", "kcal", "calorie"), agg="sum", grid=grid
            )
            heart_rate = _event_array(
                df, timestamp_col, ("heart_rate", "hr", "pulse"), agg="mean", grid=grid
            )
            sleep = self._sleep_array(sleep_lookup.get(person_id, []), grid)

            if meals is not None:
                extras["meal_series"] = meals.astype(np.float32)
            if bolus is not None:
                extras["bolus_insulin_series"] = bolus.astype(np.float32)
            if basal is not None:
                extras["basal_insulin_series"] = basal.astype(np.float32)
            if steps is not None:
                extras["activity_series"] = steps.astype(np.float32)
            if calories is not None:
                extras["cal_series"] = calories.astype(np.float32)
            if heart_rate is not None:
                extras["hr_series"] = heart_rate.astype(np.float32)
            if sleep is not None:
                extras["sleep_series"] = sleep.astype(np.float32)

            records.append(
                standardize_record(
                    glucose_z,
                    mask,
                    person_id=person_id,
                    split="train",
                    label=1,
                    dataset_name=self.name,
                    extras=extras,
                )
            )

        if not records:
            raise ValueError(f"no HUPA-UCM CGM records found under {raw_source}")

        splits = train_val_test_split(records, seed=42)
        supported_modalities = [
            "cgm",
            "meal_series",
            "bolus_insulin_series",
            "basal_insulin_series",
            "activity_series",
            "cal_series",
            "hr_series",
        ]
        if sleep_lookup:
            supported_modalities.append("sleep_series")
        metadata = DatasetMetadata(
            name=self.name,
            display_name="HUPA-UCM",
            description="Open T1D CGM + meals + insulin + activity/sleep cohort adapted to the AI-READI-compatible schema.",
            raw_source=raw_source,
            source_dir=self.default_output_dir,
            supported_modalities=supported_modalities,
            cohort_counts={
                "t1d": len({str(record["person_id"]) for record in records})
            },
            notes={
                "split_summary": summarize_splits(splits),
                "source_layout": "Preprocessed semicolon CSVs with optional Fitbit sleep summaries from Raw_Data",
                "alignment": "resampled to a 5-minute midnight-based grid so eval window [288, 576) corresponds to calendar day 2",
                "sleep_series_definition": "binary 5-minute sleep flag derived from Fitbit sleep summary intervals, including naps when available",
            },
        )
        return AdapterResult(
            train=splits["train"],
            val=splits["val"],
            test=splits["test"],
            metadata=metadata,
        )


class LoopAdapter(BaseDatasetAdapter):
    name = "loop"
    aliases = ("loop_study", "loop_observational")

    def default_prepared_dirs(self) -> list[str]:
        return [self.default_output_dir]

    def load(self, prepared_dir: str | None = None) -> AdapterResult:
        result = super().load(prepared_dir=prepared_dir)
        if result.metadata.display_name == result.metadata.name:
            result.metadata.display_name = "Loop Observational Study"
        if not result.metadata.description:
            result.metadata.description = (
                "Loop CGM cohort prepared for the Toye benchmark. "
                "Generate the split pickles with cgm_datasets/loop_etl.py."
            )
        if not result.metadata.supported_modalities:
            result.metadata.supported_modalities = ["cgm"]
        return result

    def _prepare_raw(self, raw_source: str) -> AdapterResult:
        raise NotImplementedError(
            "Loop raw conversion is handled by cgm_datasets/loop_etl.py. "
            "Run `python cgm_datasets/loop_etl.py --loop_raw_dir <raw_dir> --out_dir <prepared_dir>` "
            "and then load_dataset_splits('loop', prepared_dir=<prepared_dir>)."
        )


def build_adapters(root_dir: str) -> list[BaseDatasetAdapter]:
    from cgm_datasets.mimic_adapter import (
        MimicAbpAdapter,
        MimicHrAdapter,
        MimicIvHrAdapter,
    )

    return [
        AIReadiAdapter(root_dir),
        LoopAdapter(root_dir),
        ShanghaiAdapter(root_dir),
        HupaUcmAdapter(root_dir),
        OhioT1DMAdapter(root_dir),
        MimicAbpAdapter(root_dir),
        MimicHrAdapter(root_dir),
        MimicIvHrAdapter(root_dir),
    ]
