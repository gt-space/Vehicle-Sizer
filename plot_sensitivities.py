# This file can be modified to plto whatever you want
from pathlib import Path
from sensitivity import SweepResults



RESULTS_DIR = Path(__file__).resolve().parent / 'outputs/sensitivity/flight_pressure_fed_regulator'


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
)


results.plot_history(
    y='thrust_N',
    group_by='chamber_pressure',
    title='Thrust history across chamber pressure sweep',
    xlabel='Flight time (s)',
    ylabel='Thrust (N)',
    save=True,
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