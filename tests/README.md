# Tests

Checks component equations, conservation, events, geometry, flight coupling, plots, and optimizer behavior.

Inputs: fixtures, project configs, and lookup files for integration tests. Outputs: pytest results and temporary test artifacts.

From the project root, using the [project environment](../README.md):

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest tests/test_section_mass_inputs.py -q
```

Native fluid tests require `sundials4py`; table integration tests require the referenced HDF5 files.
