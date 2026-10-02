"""Per-signal clinical semantics for MIMIC vital-sign imputation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SignalProfile:
    name: str  # profile id, e.g. "abp_mean" | "hr"
    channel: str  # source pickle timeseries key ("abp" | "hr")
    channel_index: int  # column within that key (abp mean = 2, hr = 0)
    units: str  # "mmHg" | "bpm"
    nmar_lo: float  # clinical low (native units) for NMAR value-triggered gaps
    nmar_hi: float  # clinical high (native units)
    mar_covariate: str  # mm slot name whose high values drive MAR gaps
    cond_vitals: list[str]  # mm slot names used as multimodal conditioning
    normal_lo: float  # downstream "in-range" band low (native units)
    normal_hi: float  # downstream "in-range" band high (native units)


# MIMIC vital name -> canonical mm slot name (must exist in modality_spec.TS_MODALITIES).
# rr maps to the existing "resp" slot (respiratory rate); spo2/abp are added slots.
VITAL_TO_MM = {"abp": "abp", "hr": "hr", "rr": "resp", "spo2": "spo2"}

# vital -> (source pickle timeseries key, channel index within that key)
VITAL_SOURCE = {
    "abp": ("abp", 2),
    "hr": ("hr", 0),
    "rr": ("rr", 0),
    "spo2": ("spo2", 0),
}

PROFILES: dict[str, SignalProfile] = {
    "abp_mean": SignalProfile(
        name="abp_mean",
        channel="abp",
        channel_index=2,
        units="mmHg",
        nmar_lo=65.0,
        nmar_hi=100.0,
        mar_covariate="hr",
        cond_vitals=["hr", "resp", "spo2"],
        normal_lo=70.0,
        normal_hi=100.0,
    ),
    "hr": SignalProfile(
        name="hr",
        channel="hr",
        channel_index=0,
        units="bpm",
        nmar_lo=50.0,
        nmar_hi=100.0,
        mar_covariate="spo2",
        cond_vitals=["abp", "resp", "spo2"],
        normal_lo=60.0,
        normal_hi=100.0,
    ),
    # MIMIC-IV hourly HR: no 5-min ABP available, condition only on hourly vitals.
    "hr_hourly": SignalProfile(
        name="hr_hourly",
        channel="hr",
        channel_index=0,
        units="bpm",
        nmar_lo=50.0,
        nmar_hi=100.0,
        mar_covariate="spo2",
        cond_vitals=["resp", "spo2"],
        normal_lo=60.0,
        normal_hi=100.0,
    ),
}
