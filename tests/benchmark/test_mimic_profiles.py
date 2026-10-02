# tests/benchmark/test_mimic_profiles.py
from cgm_datasets.mimic_profiles import PROFILES, VITAL_SOURCE, VITAL_TO_MM


def test_profiles_have_required_fields():
    for key in ("abp_mean", "hr"):
        p = PROFILES[key]
        assert p.name == key
        assert p.nmar_lo < p.nmar_hi  # low threshold below high
        assert p.mar_covariate in VITAL_TO_MM.values()
        assert len(p.cond_vitals) >= 2  # at least two conditioning vitals
        assert p.channel in VITAL_SOURCE  # source channel resolvable
    # channel index must agree with the VITAL_SOURCE map so the two overlapping
    # sources of the column index cannot silently drift apart.
    for p in PROFILES.values():
        assert VITAL_SOURCE[p.channel] == (p.channel, p.channel_index)


def test_hr_hourly_conditions_on_hourly_vitals_only():
    # hourly HR has no 5-min ABP, so it conditions only on the hourly vitals.
    p = PROFILES["hr_hourly"]
    assert p.cond_vitals == ["resp", "spo2"]
    assert len(p.cond_vitals) == 2


def test_abp_uses_mean_channel():
    # ABP mean is column index 2 of the (T,3) abp array (sys, dias, mean)
    assert PROFILES["abp_mean"].channel == "abp"
    assert PROFILES["abp_mean"].channel_index == 2
    assert PROFILES["hr"].channel == "hr"
    assert PROFILES["hr"].channel_index == 0


def test_modality_spec_has_mimic_slots():
    from cgm_datasets.multimodal.modality_spec import TS_MODALITIES, ts_width

    assert TS_MODALITIES["spo2"] == ("mm_spo2", 1)
    assert TS_MODALITIES["abp"] == ("mm_abp", 1)
    # abp-target conditioning (hr,resp,spo2): each 1 value + 1 presence -> K=6
    assert ts_width(["hr", "resp", "spo2"]) == 6
