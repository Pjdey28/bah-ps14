import torch
import torch.nn as nn

class Attention(nn.Module):

    def __init__(self,hidden_size):

        super().__init__()

        self.attn=nn.Linear(hidden_size*2,1)

    def forward(self,x):

        weights=torch.softmax(

            self.attn(x),

            dim=1

        )

        context=torch.sum(

            weights*x,

            dim=1

        )

        return context,weights


class LSTMForecast(nn.Module):

    def __init__(self,input_size):

        super().__init__()

        self.lstm=nn.LSTM(

            input_size=input_size,

            hidden_size=128,

            num_layers=2,

            batch_first=True,

            dropout=0.3,

            bidirectional=True

        )

        self.attention=Attention(128)

        self.norm=nn.LayerNorm(256)

        self.dropout=nn.Dropout(0.3)

        self.fc1=nn.Linear(
            256,
            128
        )

        self.relu=nn.ReLU()

        self.fc2=nn.Linear(
            128,
            64
        )

        self.fc3=nn.Linear(
            64,3
        )

    def forward(self,x):

        out,_=self.lstm(x)

        context,weights=self.attention(out)

        context=self.norm(context)

        context=self.dropout(context)

        context=self.fc1(context)

        context=self.relu(context)

        context=self.dropout(context)

        context=self.fc2(context)

        context=self.relu(context)

        output=self.fc3(context)

        return output