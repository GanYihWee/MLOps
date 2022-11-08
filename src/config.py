from pathlib import Path


# print('getcwd:      ', os.getcwd())
# print('__file__:    ', (Path(__file__).parent.parent / "assets"))

class Config:
    RANDOM_SEED = 42
    ASSETS_PATH = (Path(__file__).parent.parent / "assets")
    ORIGINAL_DATASET_PATH = ASSETS_PATH / 'original_data' / 'Transaction_dataset.csv'
    TRAIN_TEST_PATH = ASSETS_PATH / 'train_test'
    MODEL_PATH = ASSETS_PATH /'models'
    PROCESSED_DATASET_PATH = ASSETS_PATH /'processed'
    PREDICTION_PATH = ASSETS_PATH /'predictions'
    LOG_PATH = ASSETS_PATH / 'logs'
    FEATURE_PATH = ASSETS_PATH / 'features'
    METRICS_PATH = ASSETS_PATH / 'metrics.json'
    