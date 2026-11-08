Test period 2026-08-03 to 2026-10-31 (held out from fitting and tuning).
MAE and RMSE in MW, MAPE in %. Skill = 1 - MAE / MAE of the reference baseline.

| model | CV MAE (mean +- std over 5 folds) | CV skill vs naive_week | test MAE | test RMSE | test MAPE | test skill vs naive_week |
|---|---|---|---|---|---|---|
| naive_day | 1,857 +- 80 | -0.680 +- 0.383 | 1,765 | 2,313 | 9.88 | -0.449 |
| naive_week | 1,152 +- 263 | +0.000 +- 0.000 | 1,218 | 1,618 | 7.17 | +0.000 |
| naive_mean4w | 1,183 +- 362 | -0.037 +- 0.228 | 1,489 | 1,917 | 8.70 | -0.223 |
| lgbm | 677 +- 134 | +0.406 +- 0.076 | 1,087 | 1,461 | 6.48 | +0.108 |
| lgbm_resid | 704 +- 135 | +0.380 +- 0.084 | 1,009 | 1,363 | 6.01 | +0.171 |
| mlp | 667 +- 98 | +0.410 +- 0.072 | 989 | 1,319 | 5.85 | +0.188 |
