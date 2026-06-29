import joblib
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

class XGBoostModel:

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

        Path("artifacts/models/xgboost").mkdir(
            parents=True,
            exist_ok=True
        )

        for target in self.targets:

            print(f"\nTraining XGBoost : {target}")

            y_train=train_df[target]

            y_test=test_df[target]

            model=XGBRegressor(

                n_estimators=600,

                learning_rate=0.03,

                max_depth=8,

                subsample=0.8,

                colsample_bytree=0.8,

                objective="reg:squarederror",

                random_state=42,

                n_jobs=-1

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

                f"artifacts/models/xgboost/{target}.pkl"

            )

        return results,predictions