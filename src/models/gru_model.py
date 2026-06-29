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


class ResidualBlock(nn.Module):

    def __init__(self,size):

        super().__init__()

        self.block=nn.Sequential(

            nn.Linear(size,size),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(size,size)

        )

        self.norm=nn.LayerNorm(size)

    def forward(self,x):

        return self.norm(

            x+self.block(x)

        )


class GRUForecast(nn.Module):

    def __init__(self,input_size):

        super().__init__()

        self.gru=nn.GRU(

            input_size=input_size,

            hidden_size=128,

            num_layers=2,

            batch_first=True,

            dropout=0.3,

            bidirectional=True

        )

        self.attention=Attention(128)

        self.residual=ResidualBlock(256)

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

            64,

            3

        )

    def forward(self,x):

        out,_=self.gru(x)

        context,_=self.attention(out)

        context=self.residual(context)

        context=self.dropout(context)

        context=self.fc1(context)

        context=self.relu(context)

        context=self.dropout(context)

        context=self.fc2(context)

        context=self.relu(context)

        output=self.fc3(context)

        return output