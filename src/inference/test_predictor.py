from src.models.dataset import DatasetLoader
from src.inference.predictor import SolarStormPredictor


loader = DatasetLoader()

_, test_df = loader.train_test_split()

predictor = SolarStormPredictor()

predictions = predictor.predict(test_df)

print(predictions.head())

print()

print(predictions.shape)