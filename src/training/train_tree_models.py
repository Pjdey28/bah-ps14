from src.models.dataset import DatasetLoader
from src.models.random_forest import RandomForestModel
from src.models.lightgbm_model import LightGBMModel
from src.models.xgboost_model import XGBoostModel
from src.config import SEQUENCE_LENGTH
class TreeModelTrainer:

    def __init__(self):

        self.loader=DatasetLoader()

        self.rf=RandomForestModel()

        self.lgbm=LightGBMModel()

        self.xgb=XGBoostModel()

    def train(self):

        train_df,test_df=self.loader.train_test_split()

        prediction_dict={}

        metrics={}

        print("\n==============================")
        print("Training Random Forest")
        print("==============================")

        rf_metrics,rf_predictions=self.rf.train(
            train_df,
            test_df
        )

        prediction_dict["RandomForest"]=rf_predictions

        metrics["RandomForest"]=rf_metrics

        print("\n==============================")
        print("Training LightGBM")
        print("==============================")

        lgbm_metrics,lgbm_predictions=self.lgbm.train(
            train_df,
            test_df
        )

        prediction_dict["LightGBM"]=lgbm_predictions

        metrics["LightGBM"]=lgbm_metrics

        print("\n==============================")
        print("Training XGBoost")
        print("==============================")

        xgb_metrics,xgb_predictions=self.xgb.train(
            train_df,
            test_df
        )

        prediction_dict["XGBoost"]=xgb_predictions

        metrics["XGBoost"]=xgb_metrics
        for model_name in prediction_dict:

            for target in prediction_dict[model_name]:

                prediction_dict[model_name][target] = prediction_dict[model_name][target][SEQUENCE_LENGTH:]

        targets={

            "target_30min":test_df["target_30min"].values[SEQUENCE_LENGTH:],

            "target_6hr":test_df["target_6hr"].values[SEQUENCE_LENGTH:],

            "target_12hr":test_df["target_12hr"].values[SEQUENCE_LENGTH:]
        }

        return prediction_dict,targets,metrics