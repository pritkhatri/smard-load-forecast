# Model card: smardfc day-ahead load model

**Task.** Forecast the hourly electricity load (realised consumption, MW) of the Amprion
control zone for the next local day, issued at 12:00 on the previous day.

**Served model.** LightGBM regressor (`lgbm`), 500 trees, parameters set by hand and
never tuned (`src/smardfc/models.py`), trained on all hourly data from 2022-01-30 (after a 28-day lag
warm-up) to 2026-10-31.

**Training data.** SMARD "Realisierter Stromverbrauch" for the Amprion zone, hourly,
2022-01-01 to 2026-10-31. Public data from the Bundesnetzagentur (SMARD.de).

**Inputs.** Load history up to the cut-off and calendar information (NRW public holidays).
No weather, no prices, no generation data.

**Evaluation.** Five expanding-window CV folds and a held-out final 90 days (2026-08-03 to
2026-10-31). For this model: CV MAE 677 +- 134 MW (5 folds), test MAE 1,087 MW, test MAPE
6.48%, test skill 0.108 against the same-hour-last-week baseline. The reported test numbers
come from a model fitted before the test period; the served model is also fitted on it.
The same protocol on 2025-08-03 to 2025-10-31 (no break) gives test MAE 545 MW and skill 0.225.
`lgbm` was selected by CV MAE, but after the test results had been seen. Details and the full
table are in the README and `results/`.

**Intended use.** Portfolio and demonstration of a leakage-safe forecasting pipeline.

**Not intended for.** Trading, dispatch, grid operation or any decision with financial or
safety consequences. It has not been compared with operational TSO forecasts.

**Known weaknesses.**
- Large errors when load deviates from the recent weekly pattern for reasons the features
  cannot see. In August 2026 it was no better than the weekly naive baseline.
- No weather input; extreme heat or cold is not anticipated.
- Holidays: NRW public holidays only, although the load is for the whole Amprion control area
  (Epiphany, Assumption Day and Reformation Day in other states are missed). The test period
  contains one NRW holiday (a Saturday).
- Test period includes an unexplained August 2026 level shift in the Amprion series (see README).
- Other control zones, other years and quarter-hourly resolution were not tested.
- The API forecasts only for dates inside the stored data range and does not fetch new data.
