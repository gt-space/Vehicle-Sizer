# Propulsion templates

Define fluid-network connections for pressure-fed, pump-fed, and Vespula configurations.

Inputs:

- `circuits`: fluid roles and tank references.
- `nodes`: tanks, junctions, combustor, and boundary nodes.
- `branches`: losses, pumps, controllers, switches, reliefs, and nozzle connections.

`{config: path.to.value}` reads a flight-config value. A branch's `parameters` can reference a config block. `P0` sets design pressure directly or through another node's pressure plus a rise/drop. Tank ports distinguish liquid and ullage flow.

A supplied `CdA` fixes a restriction area; eligible omitted areas are sized from design flow and pressure drop. Switch and relief areas are explicit inputs. A switch starts at `CdA_initial` and permanently changes to `CdA_switch` when its upstream (`from`) node absolute pressure reaches `switch_pressure` in `switch_direction` (`rising` or `falling`). If the condition is already met at initialization, it switches immediately.

Select a template in the flight config:

```yaml
prop_system:
  template: Configs/templates/pressure_fed_regulator.yaml
```

Run that flight config with `main.py`. The template produces the network used by vehicle sizing and fluid simulation; it is not a standalone run config.
