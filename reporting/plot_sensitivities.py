r"""Plot saved sensitivity results without running simulations.

Examples:
    python -m reporting.plot_sensitivities outputs/sensitivity/my_sweep --list-fields
    python -m reporting.plot_sensitivities outputs/sensitivity/my_sweep \
        --x tanks.press_tank.volume --y apogee --output /tmp/sweep.png

Use sensitivity.SweepResults directly for the full plotting API.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sensitivity import SweepResults


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('results', type=Path, help='Saved sweep directory')
    parser.add_argument('--list-fields', action='store_true', help='List saved scalar and history fields')
    parser.add_argument('--mode', choices=('scalar', 'history', 'derivative', 'history-derivative'),
                        default='scalar', help='Plot type (default: scalar)')
    parser.add_argument('--x', help='Scalar x field')
    parser.add_argument('--y', help='Scalar or history y field')
    parser.add_argument('--y2', help='Scalar field for the right-hand y axis')
    parser.add_argument('--parameter', help='Sweep parameter for history grouping or derivatives')
    parser.add_argument('--filter', action='append', default=[], metavar='PARAMETER=VALUE',
                        help='Fix another sweep parameter; repeat for full-grid results')
    parser.add_argument('--title')
    parser.add_argument('--xlabel')
    parser.add_argument('--ylabel')
    parser.add_argument('--y2label')
    parser.add_argument('--xscale', type=float, default=1.0, help='Divide displayed x values by this number')
    parser.add_argument('--yscale', type=float, default=1.0, help='Divide displayed y values by this number')
    parser.add_argument('--output', type=Path, help='Output image path (default: RESULTS/plots/)')
    parser.add_argument('--show', action='store_true', help='Open the Matplotlib window')
    args = parser.parse_args(argv)

    if not args.show:
        import matplotlib
        matplotlib.use('Agg')
    results = SweepResults.load(args.results)
    if args.list_fields:
        print('Scalar fields:\n' + '\n'.join(results.fields()))
        if (args.results / 'histories.h5').exists():
            print('\nHistory fields:\n' + '\n'.join(results.history_fields()))
        return
    if not args.y:
        parser.error('--y is required for plotting')
    if args.mode in ('scalar', 'derivative') and not args.x:
        parser.error('--x is required for scalar and derivative plots')
    if args.mode != 'scalar' and not args.parameter:
        parser.error('--parameter is required for history and derivative plots')
    if args.mode != 'scalar' and (args.y2 or args.y2label):
        parser.error('--y2 and --y2label apply to scalar plots')

    import yaml
    filters = {}
    for entry in args.filter:
        key, separator, value = entry.partition('=')
        if not key or not separator or not value:
            parser.error('--filter must be PARAMETER=VALUE')
        filters[key] = yaml.safe_load(value)
    options = dict(filters=filters, title=args.title, xlabel=args.xlabel,
                   ylabel=args.ylabel, xscale=args.xscale, yscale=args.yscale,
                   save=args.output if args.output is not None else True, show=args.show)
    if args.mode == 'scalar':
        figure = results.plot(args.x, args.y, y2=args.y2, y2label=args.y2label, **options)
    elif args.mode == 'history':
        figure = results.plot_history(args.y, group_by=args.parameter, **options)
    elif args.mode == 'derivative':
        figure = results.plot_derivative(args.x, args.y, wrt=args.parameter, **options)
    else:
        figure = results.plot_history_derivative(args.y, wrt=args.parameter, **options)
    import matplotlib.pyplot as plt
    plt.close(figure)


if __name__ == '__main__':
    main()
