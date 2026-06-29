import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):

    def __init__(self,d_model,max_len=5000):

        super().__init__()

        pe=torch.zeros(max_len,d_model)

        position=torch.arange(
            0,
            max_len,
            dtype=torch.float
        ).unsqueeze(1)

        div_term=torch.exp(

            torch.arange(
                0,
                d_model,
                2
            ).float()

            *(-math.log(10000.0)/d_model)

        )

        pe[:,0::2]=torch.sin(

            position*div_term

        )

        pe[:,1::2]=torch.cos(

            position*div_term

        )

        pe=pe.unsqueeze(0)

        self.register_buffer(

            "pe",

            pe

        )

    def forward(self,x):

        x=x+self.pe[:,:x.size(1)]

        return x


class TransformerForecast(nn.Module):

    def __init__(

        self,

        input_size,

        d_model=128,

        nhead=8,

        num_layers=4,

        dropout=0.2

    ):

        super().__init__()

        self.embedding=nn.Linear(

            input_size,

            d_model

        )

        self.position=PositionalEncoding(

            d_model

        )

        encoder_layer=nn.TransformerEncoderLayer(

            d_model=d_model,

            nhead=nhead,

            dim_feedforward=512,

            dropout=dropout,

            batch_first=True,

            activation="gelu"

        )

        self.encoder=nn.TransformerEncoder(

            encoder_layer,

            num_layers=num_layers

        )

        self.norm=nn.LayerNorm(

            d_model

        )

        self.dropout=nn.Dropout(

            dropout

        )

        self.head=nn.Sequential(

            nn.Linear(

                d_model,

                128

            ),

            nn.GELU(),

            nn.Dropout(

                dropout

            ),

            nn.Linear(

                128,

                64

            ),

            nn.GELU(),

            nn.Linear(

                64,

                3

            )

        )

    def forward(self,x):

        x=self.embedding(x)

        x=self.position(x)

        x=self.encoder(x)

        x=x.mean(dim=1)

        x=self.norm(x)

        x=self.dropout(x)

        x=self.head(x)

        return x