import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

from src.config import MODELS
from src.config import SHAP_DIR
from src.models.dataset import DatasetLoader


class SHAPAnalysis:

    def __init__(self):

        SHAP_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        self.loader = DatasetLoader()

    def load_data(self):

        train_df, test_df = self.loader.train_test_split()

        drop = [

            "time",

            "target_30min",

            "target_6hr",

            "target_12hr"

        ]

        features = [

            c for c in train_df.columns

            if c not in drop

        ]

        return (

            test_df[features],

            features

        )

    def analyse_lightgbm(

        self,

        target="target_30min"

    ):

        X, feature_names = self.load_data()

        model = joblib.load(

            MODELS /

            "lightgbm" /

            f"{target}.pkl"

        )

        explainer = shap.TreeExplainer(

            model

        )

        sample = X.sample(

            min(

                1000,

                len(X)

            ),

            random_state=42

        )

        shap_values = explainer.shap_values(

            sample

        )

        plt.figure()

        shap.summary_plot(

            shap_values,

            sample,

            show=False

        )

        plt.tight_layout()

        plt.savefig(

            SHAP_DIR /

            "shap_summary.png",

            dpi=300

        )

        plt.close()

        plt.figure()

        shap.summary_plot(

            shap_values,

            sample,

            plot_type="bar",

            show=False

        )

        plt.tight_layout()

        plt.savefig(

            SHAP_DIR /

            "shap_bar.png",

            dpi=300

        )

        plt.close()

        importance = pd.DataFrame({

            "Feature": feature_names,

            "Importance":

            abs(shap_values).mean(axis=0)

        })

        importance = importance.sort_values(

            "Importance",

            ascending=False

        )

        importance.to_csv(

            SHAP_DIR /

            "shap_importance.csv",

            index=False

        )

        return importance
    def rf_importance(

    self,

    target="target_30min"

):

        model = joblib.load(

            MODELS /

            "random_forest" /

            f"{target}.pkl"

        )

        _, features = self.load_data()

        imp = pd.DataFrame({

            "Feature": features,

            "Importance":

            model.feature_importances_

        })

        imp = imp.sort_values(

            "Importance",

            ascending=False

        )

        imp.to_csv(

            SHAP_DIR /

            "rf_importance.csv",

            index=False

        )

        return imp
    def xgb_importance(

    self,

    target="target_30min"

):

        model = joblib.load(

            MODELS /

            "xgboost" /

            f"{target}.pkl"

        )

        _, features = self.load_data()

        imp = pd.DataFrame({

            "Feature": features,

            "Importance":

            model.feature_importances_

        })

        imp = imp.sort_values(

            "Importance",

            ascending=False

        )

        imp.to_csv(

            SHAP_DIR /

            "xgb_importance.csv",

            index=False

        )

        return imp
   
if __name__ == "__main__":

    analysis = SHAPAnalysis()

    print("Generating SHAP...")

    analysis.analyse_lightgbm()

    print("Generating RF importance...")

    analysis.rf_importance()

    print("Generating XGBoost importance...")

    analysis.xgb_importance()

    print("Done.")