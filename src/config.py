"""Configuration settings for the project."""

from pathlib import Path

# Random seed for reproducibility
RANDOM_SEED = 42

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EXPERIMENTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model parameters
TEST_SIZE = 0.2
VAL_SIZE = 0.15

# Target variable
TARGET_COLUMN = "Watch time(Minutes)"

# Metric to optimize
PRIMARY_METRIC = "RMSE"
SECONDARY_METRICS = ["MAE", "R2"]

# Cross-validation
N_CROSS_VAL_FOLDS = 5

# Hyperparameter tuning
HYPERPARAM_N_ITER = 30  # for RandomizedSearchCV

# Feature engineering
FEATURE_ENGINEERING_ENABLED = True

# Dimensionality reduction
PCA_VARIANCE_THRESHOLD = 0.95
FEATURE_SELECTION_N_FEATURES = 10
