from encoder.hyperparameters import *
from torch import nn
import numpy as np
import torch

# Probably going to want to store these somewhere else

class SpeakerEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # TODO: The paper mentions that they use projection as referenced in https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43905.pdf
        # I'll need to create a custom LSTM that does this, but want to at least get some initial training done before I start messing aroudn with it
        self.lstm = nn.LSTM(input_size=mel_n_channels,
                            hidden_size=hidden_nodes,
                            num_layers=num_layers,
                            batch_first=True)

        self.linear = nn.Linear(in_features=hidden_nodes,
                                out_features=projection_size)

        self.relu = nn.ReLU()
    
    def forward(self, utterances):
        # We only care about the hidden state of the lstm
        _, (hidden, _) = self.lstm(utterances)

        # Grab the last hidden state
        embeddings_raw = self.relu(self.linear(hidden[-1]))

        #L2 normalize the output
        embeddings = embeddings_raw / torch.norm(embeddings_raw, dim=1, keepdim=True)

        return embeddings
    
    #TODO Implement loss as defined in paper