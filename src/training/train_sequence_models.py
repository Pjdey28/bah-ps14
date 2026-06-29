import torch

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from src.features.sequence_builder import SequenceBuilder

from src.models.base_sequence_model import BaseSequenceTrainer
from src.models.lstm_model import LSTMForecast
from src.models.gru_model import GRUForecast
from src.models.transformer_model import TransformerForecast


class SequenceTrainer:

    def __init__(self):

        self.builder=SequenceBuilder()

    def train_single(self,model_class):

        X_train,Y_train,X_test,Y_test,_= self.builder.build()

        train_loader=DataLoader(

            TensorDataset(

                torch.FloatTensor(X_train),

                torch.FloatTensor(Y_train)

            ),

            batch_size=256,

            shuffle=True

        )

        test_loader=DataLoader(

            TensorDataset(

                torch.FloatTensor(X_test),

                torch.FloatTensor(Y_test)

            ),

            batch_size=256,

            shuffle=False

        )

        model=model_class(

            input_size=X_train.shape[2]

        )

        trainer=BaseSequenceTrainer(

            model=model,

            epochs=40,

            learning_rate=1e-3,

            patience=6

        )

        trainer.train(

            train_loader,

            test_loader

        )

        metrics=trainer.evaluate(

            test_loader

        )

        pred,true=trainer.predict(

            test_loader

        )

        return metrics,pred,true

    def train(self):

        prediction_dict={}

        metrics={}

        targets={}

        m,p,t=self.train_single(

            LSTMForecast

        )

        prediction_dict["LSTM"]=p

        metrics["LSTM"]=m

        m,p,t=self.train_single(

            GRUForecast

        )

        prediction_dict["GRU"]=p

        metrics["GRU"]=m

        m,p,t=self.train_single(

            TransformerForecast

        )

        prediction_dict["Transformer"]=p

        metrics["Transformer"]=m

        targets=t

        return prediction_dict,targets,metrics