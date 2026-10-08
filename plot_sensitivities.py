from pathlib import Path

from sensitivity import SweepResults


# ---- TEST SETTINGS ---------------------------------------------------------
RESULTS_DIR = Path(__file__).resolve().parent / 'outputs/sensitivity/flight_pressure_fed_regulator'
# ---------------------------------------------------------------------------


results = SweepResults.load(RESULTS_DIR)

# Discover available variables before choosing what to plot.
#print('Scalar fields:', results.fields())
#print('Flight-history fields:', results.history_fields())

results.plot(
    x='chamber_pressure',
    y='apogee',
    title='Apogee versus chamber pressure',
    xlabel='Design chamber pressure (Pa)',
    ylabel='Apogee (m)',
    kind='line',
    marker='o',
    save=True,
    filename='apogee_vs_chamber_pressure',
    file_format='png',
)
'''
results.plot_history(
    y='thrust_N',
    group_by='chamber_pressure',
    title='Thrust history across chamber pressure sweep',
    xlabel='Flight time (s)',
    ylabel='Thrust (N)',
    figsize=(10, 5.5),
    active_only=True,             # Automatically crop the long zero-thrust tail
    colorbar=True,               # Continuous color scale replaces 31 legend entries
    cmap='viridis',
    color_scale=1e6,             # Show swept Pc values as MPa, not Pa
    colorbar_label='Chamber pressure (MPa)',
    linewidth=1.6,
    save=True,
    filename='thrust_vs_time',
    file_format='png',           # change to 'jpeg', 'svg', 'pdf', etc.
)

results.plot_derivative(
    x='chamber_pressure',
    y='apogee',
    wrt='chamber_pressure',
    title='Apogee sensitivity to chamber pressure',
    xlabel='Chamber pressure (Pa)',
    ylabel='d(Apogee) / d(Pc) (m/Pa)',
    save=True,
)

results.plot_history_derivative(
    y='altitude_m',
    wrt='chamber_pressure',
    title='Altitude sensitivity during flight',
    xlabel='Flight time (s)',
    ylabel='d(Altitude) / d(Pc) (m/Pa)',
    at=2000000,
    save=True,
)
'''

# Example: plot without saving (the Matplotlib window will appear).
# results.plot(
#     x='burn_duration', y='apogee',
#     xlabel='Burn duration (s)', ylabel='Apogee (m)',
#     title='Apogee versus burn duration', save=False, show=True,
# )

# Example: full-grid plots need filters for other enabled parameters.
# results.plot(
#     x='chamber_pressure', y='apogee',
#     filters={'cf_efficiency': 0.90},
#     title='Apogee at 90% Cf efficiency', save=True,
# )
