Test period 2025-08-03 to 2025-10-31 (held out from fitting and tuning).
MAE and RMSE in MW, MAPE in %. Skill = 1 - MAE / MAE of the reference baseline.

| model | CV MAE (mean +- std over 5 folds) | CV skill vs naive_week | test MAE | test RMSE | test MAPE | test skill vs naive_week |
|---|---|---|---|---|---|---|
| naive_day | 1,804 +- 125 | -0.722 +- 0.312 | 1,819 | 2,490 | 9.98 | -1.588 |
| naive_week | 1,078 +- 221 | +0.000 +- 0.000 | 703 | 984 | 3.81 | +0.000 |
| naive_mean4w | 1,139 +- 367 | -0.051 +- 0.212 | 739 | 948 | 3.94 | -0.051 |
| lgbm | 624 +- 131 | +0.410 +- 0.142 | 545 | 712 | 2.98 | +0.225 |
| lgbm_resid | 638 +- 128 | +0.396 +- 0.144 | 567 | 751 | 3.10 | +0.194 |
| mlp | 644 +- 119 | +0.395 +- 0.104 | 515 | 660 | 2.81 | +0.267 |
