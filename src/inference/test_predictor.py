from src.models.dataset import DatasetLoader
from src.inference.predictor import SolarStormPredictor
from src.config import ARTIFACTS

loader = DatasetLoader()

_, test_df = loader.train_test_split()

predictor = SolarStormPredictor()

predictions = predictor.predict(test_df)

predictions.to_csv(ARTIFACTS / "predictions.csv", index=False)