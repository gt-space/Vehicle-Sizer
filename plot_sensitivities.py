from pathlib import Path

from sensitivity import SweepResults


# ---- TEST SETTINGS ---------------------------------------------------------
RESULTS_DIR = Path(__file__).resolve().parent / 'outputs/sensitivity/flight_pressure_fed_regulator'
# ---------------------------------------------------------------------------


results = SweepResults.load(RESULTS_DIR)

# Discover available variables before choosing what to plot.
#print('Scalar fields:', results.fields())
#print('Flight-history fields:', results.history_fields())

'''
results.plot(
    x='weld_allowable',
    y='apogee',
    y2={
        'design_summary.tank.fuel_tank.shell_mass': 'Fuel Tank',
        'design_summary.tank.ox_tank.shell_mass': 'LOX Tank',
    },
    title='Apogee and Tank Mass versus Weld Allowable',
    xlabel='Weld Allowable (ksi)',
    ylabel='Apogee (km)',
    y2label='Tank Shell Mass (kg)',
    xscale=6.895e6,
    yscale=1e3,
    kind='line',
    marker=None,
    show=True,         # Display the plot
    save=True,         # Also save to file
    filename='apogee_and_tank_mass',
    file_format='png',
)
'''

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




print('Available COPV outputs:', results.fields('press_tank'))

results.plot(
    x='tanks.press_tank.volume', y='apogee',
    title='Apogee vs COPV Volume',
    xscale=0.001, yscale=1000,
    xlabel='COPV Volume (L)', ylabel='Apogee (km)',
    marker=None, save=True, filename='copv_volume_vs_apogee',
    file_format='png', show=False,
)
'''
eol_field = 'fluid.press_tank.eol_temperature'
if eol_field in results.fields():
    results.plot(
        x='tanks.press_tank.volume', y='apogee', y2=eol_field,
        title='Apogee and COPV EOL Temperature vs Volume',
        xscale=0.001, yscale=1000,
        xlabel='COPV Volume (L)', ylabel='Apogee (km)',
        y2label='EOL GN2 Temperature (K)', marker=None,
        save=True, filename='copv_volume_apogee_eol_temperature',
        file_format='png', show=False,
    )
else:
    print('EOL gas temperature unavailable. Record histories and confirm engine shutdown.')

results.plot(
    x='tanks.press_tank.volume',
    y={'dry_mass': 'Vehicle Dry Mass',
        'design_summary.tank.press_tank.shell_mass': 'COPV Shell Mass'},
    title='Mass vs COPV Volume',
    xscale=0.001, xlabel='COPV Volume (L)', ylabel='Mass (kg)',
    marker=None, save=True, filename='copv_volume_mass',
    file_format='png', show=False,
)'''