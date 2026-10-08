"""Example offline plots after running: python -m sensitivity Configs/sweeps/vespula_sweep.yaml"""
from pathlib import Path

from sensitivity import SweepResults


results = SweepResults.load('outputs/sensitivity/vespula')
folder = Path('outputs/sensitivity/vespula/plots')
folder.mkdir(parents=True, exist_ok=True)

print('Available scalar fields:', results.fields())
print('Available time-history fields:', results.history_fields())

# Replace chamber_pressure with the enabled continuous input for your sweep.
results.plot('chamber_pressure', 'apogee', save=folder / 'pc_vs_apogee.png')
results.plot('burn_duration', 'apogee', save=folder / 'burn_vs_apogee.png')
results.plot_history('thrust_N', group_by='chamber_pressure', save=folder / 'thrust_histories.png')
results.plot_derivative('chamber_pressure', 'apogee', wrt='chamber_pressure',
                        save=folder / 'd_apogee_d_pc.png')
results.plot_history_derivative('altitude_m', wrt='chamber_pressure',
                                save=folder / 'd_altitude_d_pc.png')
print('Plots:', folder)
