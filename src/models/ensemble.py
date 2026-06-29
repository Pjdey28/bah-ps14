import joblib
import numpy as np

from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


class StackingEnsemble:

    def __init__(self):

        self.meta_models={}

        self.targets=[

            "target_30min",

            "target_6hr",

            "target_12hr"

        ]

    def _get_prediction(self,predictions,target,index):

        if isinstance(predictions,dict):

            return predictions[target]

        return predictions[:,index]

    def train(self,prediction_dict,test_targets):

        Path(
            "artifacts/models/ensemble"
        ).mkdir(
            parents=True,
            exist_ok=True
        )

        results=[]

        for idx,target in enumerate(self.targets):

            X=[]

            for model_name in prediction_dict:

                pred=self._get_prediction(
                    prediction_dict[model_name],
                    target,
                    idx
                )

                X.append(pred)

            X=np.column_stack(X)

            y=test_targets[:,idx]

            meta=Ridge(alpha=1.0)

            meta.fit(X,y)

            pred=meta.predict(X)

            rmse=np.sqrt(
                mean_squared_error(
                    y,
                    pred
                )
            )

            mae=mean_absolute_error(
                y,
                pred
            )

            r2=r2_score(
                y,
                pred
            )

            results.append({

                "Target":target,
                "RMSE":rmse,
                "MAE":mae,
                "R2":r2

            })

            self.meta_models[target]=meta

            joblib.dump(

                meta,

                f"artifacts/models/ensemble/{target}.pkl"

            )

        return results

    def predict(self,prediction_dict):

        final_predictions={}

        for idx,target in enumerate(self.targets):

            X=[]

            for model_name in prediction_dict:

                pred=self._get_prediction(
                    prediction_dict[model_name],
                    target,
                    idx
                )

                X.append(pred)

            X=np.column_stack(X)

            final_predictions[target]=self.meta_models[target].predict(X)

        return final_predictions