"""Structural coloring for the supported network component residual equations.

No physics changes. Construct a new pattern for every session/mode layout.
Keep dense native linear algebra; reduce expensive Python residual calls.
"""
import numpy as np
from ..FluidNode import BoundaryComponent, VolumeComponent, CombustorComponent, JunctionComponent
from errors import TrialDomainError, LookupBoundsError


def pattern(network):
    own = {key: set(range(s.start, s.stop)) for key, s in network.variable_slices.items()}
    # Transport laws read thermodynamic properties, not the donor's stored
    # mass/energy or reporting geometry. Hydrostatic port pressure is separate.
    fluid = {key: {network.variable_index[f'{key}.{name}'] for name in node.variable_names
                   if name in ('P', 'T', 'T_liq', 'T_ull', 'quality')}
             for key, node in network.nodes.items()
             if isinstance(node, (VolumeComponent, BoundaryComponent))}
    pressure = {key: (own[key].copy() if isinstance(node, VolumeComponent)
                      and network.axial_specific_force else
                      {network.variable_index[f'{key}.P']} if 'P' in node.variable_names else set())
                for key, node in network.nodes.items()}
    pending = set(network.nodes) - fluid.keys()
    while pending:
        progress = False
        for key in sorted(pending):
            incoming = [(bid, b.from_node if network.directions[bid] > 0 else b.to_node)
                        for sign, bid in network.connections[key]
                        for b in (network.branches[bid],)
                        if b.active and sign * network.directions[bid] > 0]
            if not all(donor in fluid for _, donor in incoming):
                continue
            node = network.nodes[key]
            fluid[key] = own[key].copy() if isinstance(node, JunctionComponent) else set()
            if isinstance(node, CombustorComponent) and node.mode == 'combusting':
                fluid[key] |= own[key] | pressure[node.definition['ambient_node']]
            for bid, donor in incoming:
                if isinstance(node, CombustorComponent) and node.mode == 'combusting':
                    fluid[key] |= own[bid]
                elif not isinstance(node, JunctionComponent):
                    fluid[key] |= fluid[donor]
            pending.remove(key)
            progress = True
        if not progress:
            raise ValueError('Unsupported transport cycle in Jacobian pattern')
    mask = np.zeros((len(network.y), len(network.y)), dtype=bool)
    for key, node in network.nodes.items():
        deps = own[key].copy()
        for _, bid in network.connections[key]:
            b = network.branches[bid]
            if not b.active:
                continue
            deps |= own[bid]
            if isinstance(node, (VolumeComponent, JunctionComponent)):
                donor = b.from_node if network.directions[bid] > 0 else b.to_node
                deps |= fluid[donor]
        mask[network.equation_slices[key], sorted(deps)] = True
    for key, b in network.branches.items():
        deps = own[key].copy()
        if b.active:
            donor = b.from_node if network.directions[key] > 0 else b.to_node
            deps |= pressure[b.from_node] | pressure[b.to_node] | fluid[donor]
            # Nozzle equations read chamber cstar; regulator rate constraint
            # reads target thermodynamic gradients and inventory derivatives.
            if key in network.regulator_modes:
                deps |= own[b.to_node]
        mask[network.equation_slices[key], sorted(deps)] = True
    return mask


def groups(mask):
    colors, occupied = [], []
    for column in np.argsort(-mask.sum(axis=0), kind='stable'):
        for indices, rows in zip(colors, occupied):
            if not np.any(rows & mask[:, column]):
                indices.append(column)
                rows |= mask[:, column]
                break
        else:
            colors.append([column])
            occupied.append(mask[:, column].copy())
    return [np.asarray(c) for c in colors]


def install(session, network, *, verify=False):
    mask = pattern(network)
    colors = groups(mask)
    session.structured_calls = 0
    session.structured_residuals = 0
    session.structured_colors = len(colors)
    session.structured_verified = False

    def jacobian(t, cj, yvec, ypvec, rvec, matrix, userdata, tmp1, tmp2, tmp3):
        try:
            view = session.core.N_VGetNumpyArray
            y, yp, residual = view(yvec), view(ypvec), view(rvec)
            # Scale perturbations in physical units, with representable deltas.
            step = np.sqrt(np.finfo(float).eps) * np.maximum(np.abs(y), 1e-4)
            step = (y + step) - y
            jac = session.core.SUNDenseMatrix_Data(matrix)
            jac[:] = 0.
            trial = np.empty_like(y)

            def difference(indices):
                dy = np.zeros_like(y)
                dy[indices] = step[indices]
                try:
                    session.structured_residuals += 1
                    network.residual(t, y + dy, yp + cj * dy, trial)
                except (TrialDomainError, LookupBoundsError):
                    dy = -dy
                    session.structured_residuals += 1
                    network.residual(t, y + dy, yp + cj * dy, trial)
                if not np.isfinite(trial).all():
                    raise TrialDomainError('Non-finite Jacobian trial residual')
                return trial - residual, dy

            for columns in colors:
                delta, dy = difference(columns)
                for column in columns:
                    rows = mask[:, column]
                    jac[rows, column] = delta[rows] / dy[column]
            if verify and not session.structured_verified:
                dense = np.empty_like(jac)
                for column in range(len(y)):
                    delta, dy = difference([column])
                    dense[:, column] = delta / dy[column]
                scale = np.maximum(np.abs(y), 1e-4)
                denominator = np.maximum(np.max(np.abs(dense * scale), axis=1), 1e-8)
                error = np.max(np.abs((jac - dense) * scale) / denominator[:, None])
                if error > 5e-4:
                    raise AssertionError(f'Colored versus dense Jacobian relative row error {error:g}')
                session.structured_verified = True
            session.structured_calls += 1
            if not np.isfinite(jac).all():
                raise TrialDomainError('Non-finite structured Jacobian')
            return 0
        except (TrialDomainError, LookupBoundsError) as error:
            session.last_trial_error = error
            return 1
        except Exception as error:
            session.callback_error = error
            return -1

    session._structured_callback = jacobian
    session._check(session.idas.IDASetJacFn(session.mem, jacobian), 'structured Jacobian')
