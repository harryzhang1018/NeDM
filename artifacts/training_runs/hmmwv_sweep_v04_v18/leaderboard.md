# HMMWV Transformer Sweep Leaderboard

Rank metric: median XY RMSE over the fixed 20 validation rollouts, then mean XY RMSE.

| Rank | Version | Recipe | Best Val Loss | Median XY RMSE m | Mean XY RMSE m | Median Yaw RMSE rad | Epochs |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | v07 | context128_b64 | 0.039943 | 5.956 | 31.533 | 0.0800 | 80 |
| 2 | v12 | wide512_b48 | 0.029011 | 9.431 | 72.992 | 0.0813 | 80 |
| 3 | v17 | head512_b96 | 0.039451 | 9.706 | 35.016 | 0.0719 | 80 |
| 4 | v04 | long_baseline_b32 | 0.040532 | 10.167 | 15.117 | 0.0613 | 80 |
| 5 | v13 | dropout384_b64 | 0.041471 | 10.223 | 24.191 | 0.0641 | 80 |
| 6 | v09 | deeper10_b64 | 0.037051 | 10.692 | 35.411 | 0.0791 | 80 |
| 7 | v10 | wide384_b96 | 0.039595 | 12.202 | 25.943 | 0.0932 | 80 |
| 8 | v06 | context96_b96 | 0.041817 | 12.392 | 34.515 | 0.0644 | 80 |
| 9 | v08 | deeper8_b96 | 0.039730 | 12.569 | 38.280 | 0.0956 | 80 |
| 10 | v18 | wide384_context96_b48 | 0.024761 | 15.513 | 24.521 | 0.0782 | 80 |
| 11 | v11 | wide384_deep8_b64 | 0.036532 | 16.055 | 22.809 | 0.0715 | 80 |
| 12 | v05 | context64_b128 | 0.038145 | 17.468 | 22.378 | 0.0873 | 80 |
