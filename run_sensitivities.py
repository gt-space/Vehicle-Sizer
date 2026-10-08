"""
Run the optional sensitivity sweep.
Edit sweep config before running.
"""
from pathlib import Path

from sensitivity import run_file


# ---- USER SETTINGS ---------------------------------------------------------
SWEEP_CONFIG = "Configs/sweeps/vespula_sweep.yaml"
DRY_RUN = False  # True: validate and list cases without running flights
# ---------------------------------------------------------------------------


def main():
    root = Path(__file__).resolve().parent
    config_path = Path(SWEEP_CONFIG).expanduser()
    if not config_path.is_absolute():
        config_path = root / config_path

    if not config_path.is_file():
        raise FileNotFoundError(f"Sensitivity configuration not found: {config_path}")

    result = run_file(config_path, dry_run=DRY_RUN)
    if DRY_RUN:
        print("Dry run finished; no flight simulations were executed.")
    else:
        print(f"Sensitivity results saved to: {result.folder}")
        print("Open plot_sensitivities.py and click Run Python File to plot them.")
    return result


if __name__ == "__main__":
    main()
