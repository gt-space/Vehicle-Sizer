# Wind sensitivity

Heights are above launch (AGL). Apogee is MSL; downrange is signed displacement at apogee, not landing range.
Both wind components are scaled together. Cutoff cases set wind to zero above the cutoff; scaling cases retain the original profile above the selected band.
Cutoff plot uses a linear axis from 0–10 m and logarithmic spacing above 10 m. Apogee is absolute MSL altitude.
Transitions at band boundaries are abrupt. The cutoff plot connects successful samples across failed regions; markers represent successful simulations only. Raw failures and older cutoff results remain in the results files. A 100% scaling case is identical to the full-profile baseline and is reused.

| Case | Downrange (km) | Apogee (km MSL) | Status |
|---|---:|---:|---|
| full_profile | 86.756 | 123.822 | apogee |
| cutoff_0m | 0.000 | 142.947 | apogee |
| cutoff_5m | 0.000 | 142.947 | apogee |
| cutoff_10m | 0.000 | 142.946 | apogee |
| cutoff_15m | 0.000 | 142.947 | apogee |
| cutoff_20m | 5.240 | 142.889 | apogee |
| cutoff_30m | 13.569 | 142.556 | apogee |
| cutoff_40m | 20.046 | 142.089 | apogee |
| cutoff_55m | 27.618 | 141.302 | apogee |
| cutoff_75m | 35.043 | 140.265 | apogee |
| cutoff_100m | 41.426 | 139.155 | apogee |
| cutoff_150m | 49.186 | 137.549 | apogee |
| cutoff_200m | 54.153 | 136.414 | apogee |
| cutoff_300m | 60.486 | 134.705 | apogee |
| cutoff_400m | 64.180 | 133.540 | apogee |
| cutoff_550m | 67.826 | 132.359 | apogee |
| cutoff_750m | 70.801 | 131.317 | apogee |
| cutoff_1000m | 73.168 | 130.410 | apogee |
| cutoff_2000m | 77.161 | 128.748 | apogee |
| cutoff_3000m | 78.975 | 127.879 | apogee |
| cutoff_6500m | nan | nan | failed |
| cutoff_10000m | 86.525 | 123.872 | apogee |
| cutoff_30000m | 86.753 | 123.821 | apogee |
| cutoff_100000m | 86.756 | 123.822 | apogee |
| scale_1000m_90pct | 81.377 | 126.513 | apogee |
| scale_1000m_80pct | 75.644 | 129.040 | apogee |
| scale_1000m_70pct | 69.576 | 131.389 | apogee |
| scale_1000m_60pct | 63.194 | 133.548 | apogee |
| scale_1000m_50pct | 56.516 | 135.506 | apogee |
| scale_1000m_40pct | 49.557 | 137.253 | apogee |
| scale_1000m_30pct | 42.341 | 138.780 | apogee |
| scale_10000m_90pct | 80.057 | 127.182 | apogee |
| scale_10000m_80pct | 72.798 | 130.285 | apogee |
| scale_10000m_70pct | 65.014 | 133.101 | apogee |
| scale_10000m_60pct | 56.753 | 135.602 | apogee |
| scale_10000m_50pct | 48.070 | 137.764 | apogee |
| scale_10000m_40pct | 39.014 | 139.568 | apogee |
| scale_10000m_30pct | 29.644 | 140.995 | apogee |
| cutoff_15000m | 86.771 | 123.812 | apogee |
| cutoff_20000m | 86.754 | 123.820 | apogee |
| cutoff_25000m | 86.753 | 123.821 | apogee |
| cutoff_40000m | 86.755 | 123.821 | apogee |
| cutoff_50000m | 86.756 | 123.822 | apogee |
| cutoff_75000m | 86.756 | 123.822 | apogee |

These sweeps show cumulative and trajectory-dependent effects; differences between cutoff cases are not independent additive contributions from altitude layers.

Reproduce: `.venv/bin/python wind_sensitivity/run.py --workers 3`
Replot: `.venv/bin/python wind_sensitivity/run.py --plot-only`

Inputs: `config.yaml`, `wind_profile.csv`. Outputs: `results.csv`, `results.json`, PNG/SVG plots, and failure diagnostics.
