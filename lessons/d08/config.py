from pathlib import Path

# Part01: file path
D08_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = D08_DIR / "outputs"

OUTPUT_DIR.mkdir(parents = True, exist_ok = True)

# Part02: Hyperparameters
DATA_SEED = 42
MODEL_SEED = 123
LOADER_SEED = 456

BATCH_SIZE = 16
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0

TOTAL_EPOCHS = 4
INTERRUPT_EPOCHS = 2

# Part03: Output file path
DATA_PATH = OUTPUT_DIR / "d08_data.pt"
BASELINE_PATH = OUTPUT_DIR / "d08_baseline.pt"
BASELINE_JSON_PATH = OUTPUT_DIR / "d08_baseline.json"
RESUME_PATH = OUTPUT_DIR / "d08_resume.pt"
RESUMED_PATH = OUTPUT_DIR / "d08_resumed.pt"
RESUMED_JSON_PATH = OUTPUT_DIR / "d08_resumed.json"


