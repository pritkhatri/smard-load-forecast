import pandas as pd

from smardfc.cli import main
from smardfc.data import load_processed


def test_data_command_reads_csv_and_writes_processed(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    idx = pd.date_range("2022-01-01", periods=72, freq="h")
    rows = ["Datum von;Datum bis;Netzlast [MWh] Originalauflösungen"]
    for i, t in enumerate(idx):
        end = t + pd.Timedelta(hours=1)
        value = f"{20000 + i:,}".replace(",", ".") + ",5"
        rows.append(f"{t:%d.%m.%Y %H:%M};{end:%d.%m.%Y %H:%M};{value}")
    (tmp_path / "smard.csv").write_text("\n".join(rows), encoding="utf-8")
    main(["data", "--csv", "smard.csv"])
    out = capsys.readouterr().out
    assert "72 hourly values" in out and "n_gaps: 0" in out
    s = load_processed("data/processed/load.parquet")
    assert len(s) == 72 and s.iloc[0] == 20000.5
    assert (tmp_path / "data/processed/validation_report.json").exists()
