import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "aggregate_shortgap.py")


def _write(p, method, vals):
    json.dump(
        {
            "method": method,
            "label": method,
            "by_gap_min": {
                "15min": vals[0],
                "30min": vals[1],
                "45min": vals[2],
                "60min": vals[3],
            },
        },
        open(p, "w"),
    )


def test_aggregate_emits_native_and_znorm(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    _write(res / "hr_shortgap_pchip.json", "pchip", [10.0, 12.0, 14.0, 16.0])
    _write(res / "hr_shortgap_cair.json", "cair", [8.0, 9.0, 10.0, 11.0])
    out = tmp_path / "report.md"
    subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "--results_dir",
            str(res),
            "--target",
            "hr",
            "--native_std",
            "14.968",
            "--out",
            str(out),
        ],
        check=True,
    )
    txt = out.read_text()
    assert "pchip" in txt and "cair" in txt
    assert "15min" in txt or "15 min" in txt
    # native value present, and z-normalized = native / std present
    assert "8.00" in txt  # cair native 15min
    assert f"{8.0/14.968:.3f}" in txt  # cair z-normalized 15min
