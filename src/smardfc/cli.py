"""Command line interface: smardfc data | evaluate | train | serve."""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="smardfc", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("data", help="download SMARD data (or read a CSV) and validate it")
    d.add_argument("--region", default="Amprion")
    d.add_argument("--start", default="2022-01-01", help="first local date")
    d.add_argument(
        "--end", default=None, help="exclusive local date (default: start of this month)"
    )
    d.add_argument("--csv", default=None, help="use a SMARD download-centre CSV instead of the API")
    d.add_argument("--refresh", action="store_true", help="re-download the newest cached chunks")

    e = sub.add_parser("evaluate", help="cross-validation + held-out test, writes results/")
    e.add_argument("--no-nn", action="store_true", help="skip the PyTorch MLP")
    e.add_argument("--test-days", type=int, default=90)
    e.add_argument("--until", default=None, help="use only data before this local date (backtest)")
    e.add_argument("--results", default="results", help="output directory")
    e.add_argument("--purge-days", type=int, default=0, help="gap between train and scored days")

    sub.add_parser("train", help="fit the headline model on all data and save it to models/")

    s = sub.add_parser("serve", help="start the FastAPI service")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)

    a = p.parse_args(argv)
    if a.cmd == "data":
        from .pipeline import prepare_data

        s_, rep = prepare_data(a.region, a.start, a.end, a.csv, a.refresh)
        info = {k: v for k, v in rep.to_dict().items() if k != "gaps"}
        print(f"{len(s_)} hourly values, {rep.first} .. {rep.last}")
        for k, v in info.items():
            print(f"  {k}: {v}")
    elif a.cmd == "evaluate":
        from .pipeline import evaluate

        evaluate(
            use_nn=not a.no_nn, test_days=a.test_days, until=a.until,
            results=a.results, purge_days=a.purge_days,
        )  # fmt: skip
        print((Path(a.results) / "results.md").read_text())
    elif a.cmd == "train":
        from .pipeline import train

        print(f"saved {train()}")
    elif a.cmd == "serve":
        import uvicorn

        uvicorn.run("smardfc.api:app_factory", factory=True, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
