"""Canonical paths for this experiment inside the WRS workspace."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
MODEL_DIR = WORKSPACE_ROOT / "0000_models" / "pepper"
DATASETS_DIR = PROJECT_ROOT / "datasets"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
STUDIO_ROOT = PROJECT_ROOT.parent / "robot_sim_studio"
DEFAULT_CALIBRATION = WORKSPACE_ROOT / "manual_calibration_realman22.json"

# Direct script launches must also resolve the sibling WRS library.
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
