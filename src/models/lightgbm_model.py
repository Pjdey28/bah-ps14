import joblib
import numpy as np
from pathlib import Path
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from src.config import MODELS
save_dir=MODELS/"lightgbm"
class LightGBMModel:

    def __init__(self):

        self.models={}

        self.targets=[

            "target_30min",

            "target_6hr",

            "target_12hr"

        ]

    def train(self,train_df,test_df):

        drop=[

            "time",

            "source_file",
            "source_file_x",
            "source_file_y",

            "target_30min",

            "target_6hr",

            "target_12hr"

        ]

        features=[

            c for c in train_df.columns

            if c not in drop

        ]

        X_train=train_df[features]

        X_test=test_df[features]

        results=[]
        predictions={}

        save_dir.mkdir(

            parents=True,

            exist_ok=True

        )
        for target in self.targets:

            print(f"\nTraining LightGBM : {target}")

            y_train=train_df[target]

            y_test=test_df[target]

            model=LGBMRegressor(

                n_estimators=500,

                learning_rate=0.03,

                num_leaves=64,

                random_state=42

            )

            model.fit(

                X_train,

                y_train

            )

            pred=model.predict(

                X_test

            )
            predictions[target]=pred

            rmse=np.sqrt(

                mean_squared_error(

                    y_test,

                    pred

                )

            )

            mae=mean_absolute_error(

                y_test,

                pred

            )

            r2=r2_score(

                y_test,

                pred

            )

            results.append({

                "Target":target,

                "RMSE":rmse,

                "MAE":mae,

                "R2":r2

            })

            self.models[target]=model

            joblib.dump(

                model,

                save_dir/f"{target}.pkl"

            )

        return results, predictions