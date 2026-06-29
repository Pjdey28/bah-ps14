import copy
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


class BaseSequenceTrainer:

    def __init__(
        self,
        model,
        epochs=40,
        learning_rate=1e-3,
        patience=6
    ):

        self.device=torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model=model.to(self.device)

        self.epochs=epochs

        self.optimizer=torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

        self.criterion=nn.MSELoss()

        self.patience=patience

    def train(
        self,
        train_loader,
        val_loader
    ):

        best_loss=np.inf

        counter=0

        best_state=None

        for epoch in range(self.epochs):

            self.model.train()

            train_loss=0

            for X,y in train_loader:

                X=X.to(self.device)

                y=y.to(self.device)

                self.optimizer.zero_grad()

                output=self.model(X)

                loss=self.criterion(
                    output,
                    y
                )

                loss.backward()

                self.optimizer.step()

                train_loss+=loss.item()

            train_loss/=len(train_loader)

            val_loss=0

            self.model.eval()

            with torch.no_grad():

                for X,y in val_loader:

                    X=X.to(self.device)

                    y=y.to(self.device)

                    output=self.model(X)

                    loss=self.criterion(
                        output,
                        y
                    )

                    val_loss+=loss.item()

            val_loss/=len(val_loader)

            print(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Train {train_loss:.6f} | "
                f"Val {val_loss:.6f}"
            )

            if val_loss<best_loss:

                best_loss=val_loss

                counter=0

                best_state=copy.deepcopy(
                    self.model.state_dict()
                )

            else:

                counter+=1

                if counter>=self.patience:

                    print("Early stopping")

                    break

        if best_state is not None:

            self.model.load_state_dict(
                best_state
            )

    def predict(
        self,
        loader
    ):

        self.model.eval()

        preds=[]

        actual=[]

        with torch.no_grad():

            for X,y in loader:

                X=X.to(self.device)

                output=self.model(X)

                preds.append(
                    output.cpu().numpy()
                )

                actual.append(
                    y.numpy()
                )

        preds=np.vstack(preds)

        actual=np.vstack(actual)

        return preds,actual

    def evaluate(
        self,
        loader
    ):

        pred,true=self.predict(loader)

        metrics={}

        horizons=[
            "30min",
            "6hr",
            "12hr"
        ]

        for i,name in enumerate(horizons):

            rmse=np.sqrt(
                mean_squared_error(
                    true[:,i],
                    pred[:,i]
                )
            )

            mae=mean_absolute_error(
                true[:,i],
                pred[:,i]
            )

            r2=r2_score(
                true[:,i],
                pred[:,i]
            )

            metrics[name]={
                "RMSE":rmse,
                "MAE":mae,
                "R2":r2
            }

        return metrics