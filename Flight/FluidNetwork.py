from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
from scipy.optimize import root


@dataclass
class NetworkState:
    node: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    br: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class FluidNetwork:

    def __init__(self, nodes: Dict[str, Dict[str, Any]], branches: Dict[str, Dict[str, Any]]) -> None:

        self.nodes = nodes
        self.branches = branches

        #Temporary
        self.nodes = self._filter_steady_nodes(nodes)
        self.branches = self._filter_steady_branches(branches, self.nodes)

        self.state = NetworkState()

        self.incoming = {nid: [] for nid in self.nodes}
        self.outgoing = {nid: [] for nid in self.nodes}
        self._node_map()

    def _node_map(self) -> None:
        for bid, br in self.branches.items():
            n_from = br["from"]
            n_to = br["to"]

            if n_from not in self.nodes:
                raise ValueError(f"Branch '{bid}' has unknown from-node '{n_from}'")
            if n_to not in self.nodes:
                raise ValueError(f"Branch '{bid}' has unknown to-node '{n_to}'")

            self.outgoing[n_from].append(bid)
            self.incoming[n_to].append(bid)

    def resolve_network(self, 
                        bcs: Dict[str, Dict[str, Any]], 
                        controls: Dict[str, Any], 
                        liquid_props: Dict[str, Dict[str, float]]) -> Dict[str, Any]:

        unknown_nodes = self._unsolved_nodes(bcs)
        x0 = self._initial_guess(self, unknown_nodes)

        sol = root(
            lambda x: self._steady_residual(
                x=x,
                unknown_nodes=unknown_nodes,
                bcs=bcs,
                controls=controls,
                liquid_props=liquid_props,
            ),
            x0,
            method="hybr",
        )

        if not sol.success:
            raise RuntimeError(f"Network steady solve failed: {sol.message}")

        self._steady_residual(
            x=sol.x,
            unknown_nodes=unknown_nodes,
            bcs=bcs,
            controls=controls,
            liquid_props=liquid_props,
        )

        return {
            "success": True,
            "message": sol.message,
            "node": self.state.node,
            "branch": self.state.br,
            "mdot": {bid: s["mdot"] for bid, s in self.state.br.items()},
        }

    def _unsolved_nodes(self, bcs: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Filter out boundary condition nodes from root solver
        The idea here is to set up the capability for a generalized solver where the user specifies boundary nodes
        Boundary nodes represent the shift from dynamic volumes to steady state solves
        """
        state = []

        for node in self.nodes:
            nid = node["id"]
            ntype = node["type"]

            if nid in bcs:
                continue

            if ntype == "boundary_pressure":
                continue

            state.append(nid)

        return state

    def _initial_guess(self, unknown_nodes):
        """
        Assigns inital guesses for root solve, previous guess for all timesteps unless state unfilled (T=0)
        """
        x0 = []
        for nid in unknown_nodes:
            if nid in self.state.node and "P" in self.state.node[nid]:
                p0 = self.state.node[nid]["P"]
            else:
                p0 = 0 #better guess logic for inital time
            x0.append(p0)

        return np.array(x0, dtype=float)

    def _steady_residual(self,
                         x: np.ndarray,
                         unknown_nodes: List[str],
                         bcs: Dict[str, Dict[str, Any]],
                         controls: Dict[str, Any],
                         liquid_props: Dict[str, Dict[str, float]]) -> np.ndarray:

        node_state = self._build_node_state(x, unknown_nodes, bcs)
        br_state: Dict[str, Dict[str, Any]] = {}

        for br in self.branches:
            mdot, extra = self._branch_mdot(
                br=br,
                node_state=node_state,
                controls=controls,
                liquid_props=liquid_props,
            )
            br_state[br["id"]] = {"mdot": mdot, **extra}

        R = np.zeros(len(unknown_nodes), dtype=float)

        for i, nid in enumerate(unknown_nodes):
            mdot_in = sum(br_state[br["id"]]["mdot"] for br in self.incoming[nid])
            mdot_out = sum(br_state[br["id"]]["mdot"] for br in self.outgoing[nid])
            R[i] = mdot_in - mdot_out

        self.state = NetworkState(node=node_state, br=br_state)
        return R

    def _build_node_state(self,
                          x: np.ndarray,
                          unknown_nodes: List[str],
                          bcs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:

        node_state: Dict[str, Dict[str, Any]] = {}

        for nid, bc in bcs.items():
            node_state[nid] = dict(bc)

        for i, nid in enumerate(unknown_nodes):
            if nid not in node_state:
                node_state[nid] = {}
            node_state[nid]["P"] = float(x[i])

        return node_state

    def _eval_branch_mdot(
        self,
        br: Dict[str, Any],
        node_state: Dict[str, Dict[str, Any]],
        controls: Dict[str, Any],
        liquid_props: Dict[str, Dict[str, float]],
    ) -> Tuple[float, Dict[str, Any]]:

        bid = br["id"]
        btype = br["type"]
        n_from = br["from"]
        n_to = br["to"]

        P_from = node_state[n_from]["P"]
        P_to = node_state[n_to]["P"]

        valve_mult = self._branch_open_fraction(bid, controls)

        if btype in ("liquid_loss", "injector"):
            tag = self._fluid_tag_from_branch(bid)
            rho = liquid_props[tag]["rho"]
            CdA = br["CdA"]
            dP = P_from - P_to

            mdot = valve_mult * np.sign(dP) * CdA * np.sqrt(max(2.0 * rho * abs(dP), 0.0))
            return mdot, {"dP": dP, "rho": rho}

        if btype == "gas_orifice":
            dP = P_from - P_to
            rho = br.get("rho_ref", self.params.get("gas_rho_ref", 1.0))
            CdA = br["CdA"]

            # placeholder incompressible-style form
            # replace later with compressible choked/un-choked gas model
            mdot = valve_mult * np.sign(dP) * CdA * np.sqrt(max(2.0 * rho * abs(dP), 0.0))
            return mdot, {"dP": dP, "rho": rho}

        if btype == "pump":
            tag = self._fluid_tag_from_branch(bid)
            rho = liquid_props[tag]["rho"]

            pump_dP = br["dP"]
            CdA = br.get("CdA", None)

            if CdA is None:
                raise ValueError(
                    f"Pump branch '{bid}' needs a flow law as well as dP "
                    "(or you need to model the pump another way)."
                )

            dP_eff = (P_from + pump_dP) - P_to
            mdot = valve_mult * np.sign(dP_eff) * CdA * np.sqrt(max(2.0 * rho * abs(dP_eff), 0.0))
            return mdot, {"dP_eff": dP_eff, "rho": rho}

        raise ValueError(f"Unsupported branch type '{btype}' for branch '{bid}'")

    def _fluid_tag_from_branch(self, branch_id: str) -> str:

        bid = branch_id.upper()
        if bid.startswith("OX"):
            return "ox"
        if bid.startswith("FUEL"):
            return "fuel"
        return "gas"
    

    """
    Temp functions to filter out dynamic nodes, eventually should move generalize (support multiple templates) and 
    move transient solve (residual, packing functions) from propsystem to fluid network. Since it is not generalized
    the dynamic nodes (need transient solver) need to be filtered out as they are not used.
    """


    def _filter_steady_nodes(self, nodes: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        steady_nodes = {}

        for nid, node in nodes.items():
            if node.get("steady", True):
                steady_nodes[nid] = node

        return steady_nodes
    
    def _filter_steady_branches(self, branches: Dict[str, Dict[str, Any]], steady_nodes: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        steady_branches = {}

        for bid, br in branches.items():
            if not br.get("steady", True):
                continue

            n_from = br["from"]
            n_to = br["to"]

            if n_from not in steady_nodes or n_to not in steady_nodes:
                continue

            steady_branches[bid] = br

        return steady_branches