# Diagnostics

`errors.py` defines model, solver, infrastructure, and search failures and their
classification. `constraints.py` defines feasibility records, design and operating
rejections, and constraint reporting. `warnings.py` collects nonfatal cautions.

Import the specific module, for example:

```python
from diagnostics.errors import UnsupportedPhaseChangeError
from diagnostics.constraints import DesignInfeasible
from diagnostics.warnings import caution
```

Warnings remain separate from errors and do not reject an evaluation. Physical
limit violations produce feasibility records; unsupported phase transitions
produce model errors.
