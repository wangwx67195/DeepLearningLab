from pathlib import Path

MODE = "r"

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

BATCH_SIZE = 16

TOTAL_EPOCHS = 4
INTERRUPT_EPOCHS = 2

DATA_PATH = OUTPUT_DIR / "d08_data.pt"

# OUTPUT_PATH: save the checkpoint file
BASELINE_PATH = OUTPUT_DIR / "d08_baseline.pt"
INTERRUPT_PATH = OUTPUT_DIR / "d08_resume.pt"
RESUME_PATH = OUTPUT_DIR / "d08_resumed.pt"

# JSON_PATH: Save the record json file
BASELINE_JSON_PATH = OUTPUT_DIR / "d08_baseline.json"
INTERRUPT_JSON_PATH = OUTPUT_DIR / "d08_resume.json"
RESUME_JSON_PATH = OUTPUT_DIR / "d08_resumed.json"

# CKPT_PATH: Load the checkpoint file, only resume mode needs it
RESUME_CHKPT_PATH = OUTPUT_DIR / "d08_resumed.pt"

# Part03: Output file path
if MODE == "baseline":
    OUTPUT_PATH = BASELINE_PATH
    JSON_PATH = BASELINE_JSON_PATH
elif MODE == "interrupt":
    OUTPUT_PATH = INTERRUPT_PATH
    JSON_PATH = INTERRUPT_JSON_PATH
elif MODE == "resume":
    CKPT_PATH = RESUME_CHKPT_PATH
    OUTPUT_PATH = RESUME_PATH
    JSON_PATH = RESUME_JSON_PATH

