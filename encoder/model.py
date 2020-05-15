from torch import nn
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import numpy as np
import torch

from encoder.hyperparameters import *

# Probably going to want to store these somewhere else

class SpeakerEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # TODO: The paper mentions that they use projection as referenced in
        # https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43905.pdf
        # I'll need to create a custom LSTM that does this, but want to at least get some initial
        # training done before I start messing aroudn with it
        self.lstm = nn.LSTM(input_size=MEL_N_CHANNELS,
                            hidden_size=HIDDEN_NODES,
                            num_layers=NUM_LAYERS,
                            batch_first=True)

        self.linear = nn.Linear(in_features=HIDDEN_NODES,
                                out_features=PROJECTION_SIZE)

        self.relu = nn.ReLU()

        self.similarity_weight = nn.Parameter(torch.Tensor([SIMILARITY_WEIGHT_INIT]))
        self.similarity_bias = nn.Parameter(torch.Tensor([SIMILARITY_BIAS_INIT]))

        self.loss_fn = nn.CrossEntropyLoss()

    def do_gradient_ops(self):
        # Section 3. Experiments of the paper mentions that the l2-norm of gradient is clipped at 3
        # and that the (w, b) from the loss function having a smaller gradient scale of 0.01 helps
        # smooth convergence

        self.similarity_weight.grad *= 0.01
        self.similarity_bias.grad *= 0.01

        clip_grad_norm_(self.parameters(), L2_NORM_CLIP, norm_type=2)

    def forward(self, utterances):
        # We only care about the hidden state of the lstm
        _, (hidden, _) = self.lstm(utterances)

        # Grab the last hidden state
        embeddings_raw = self.relu(self.linear(hidden[-1]))

        #L2 normalize the output
        embeddings = embeddings_raw / torch.norm(embeddings_raw, dim=2, keepdim=True)

        return embeddings

    def similarity_matrix(self, embeds):
        """
        Computes the similarity matrix according the section 2.1 of GE2E.
        :param embeds: the embeddings as a tensor of shape (speakers_per_batch, 
        utterances_per_speaker, embedding_size)
        :return: the similarity matrix as a tensor of shape (speakers_per_batch,
        utterances_per_speaker, speakers_per_batch)
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]

        centroids_incl = torch.mean(embeds, dim=1, keepdim=True)
        centroids_incl = centroids_incl.clone() / torch.norm(centroids_incl, dim=2, keepdim=True)

        centroids_excl = torch.sum(embeds, dim=1, keepdim=True) - embeds
        centroids_excl /= (utterances_per_speaker - 1)
        centroids_excl = centroids_excl.clone() / torch.norm(centroids_excl, dim=2, keepdim=True)

        sim_matrix = torch.zeros(speakers_per_batch, utterances_per_speaker, speakers_per_batch)

        mask_matrix = 1 - np.eye(speakers_per_batch, dtype=np.int)

        for j in range(speakers_per_batch):
            mask = np.where(mask_matrix[j])[0] 
            # Cosine similarity of l2 normalized vectors is simply the dot product
            sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(dim=2)
            sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)

        sim_matrix = sim_matrix * self.similarity_weight + self.similarity_bias
        return sim_matrix

    def loss(self, embeds):
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]

        sim_matrix = self.similarity_matrix(embeds)
        sim_matrix = sim_matrix.reshape((speakers_per_batch * utterances_per_speaker, speakers_per_batch))

        ground_truth = np.repeat(np.arange(speakers_per_batch), utterances_per_speaker)
        target = torch.from_numpy(ground_truth).long()
        loss = self.loss_fn(sim_matrix, target)

        with torch.no_grad():
            inv_argmax = lambda i: np.eye(1, speakers_per_batch, i, dtype=np.int)[0]
            labels = np.array([inv_argmax(i) for i in ground_truth])
            preds = sim_matrix.detach().cpu().numpy()

            fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

        return loss, eer