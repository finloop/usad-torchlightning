import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from pytorch_lightning.core.lightning import LightningModule
from collections import OrderedDict


class Encoder(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(Encoder, self).__init__()
    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )
    self.rnn2 = nn.LSTM(
      input_size=self.hidden_dim,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=True
    )
  def forward(self, x):
    x = x.reshape((-1, self.seq_len, self.n_features))
    x, (_, _) = self.rnn1(x)
    x, (hidden_n, _) = self.rnn2(x)
    return hidden_n.reshape((-1, self.embedding_dim))

class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.rnn1 = nn.LSTM(
          input_size=input_dim,
          hidden_size=input_dim,
          num_layers=1,
          batch_first=True
        )
        self.rnn2 = nn.LSTM(
          input_size=input_dim,
          hidden_size=self.hidden_dim,
          num_layers=1,
          batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        x = x.repeat(self.seq_len, 1)
        x = x.reshape((-1, self.seq_len, self.input_dim))
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = x.reshape((-1, self.seq_len, self.hidden_dim))
        return self.output_layer(x)


class USADModel(LightningModule):
    def __init__(self, seq_len, n_features, embedding_dim, learning_rate=1e-3):
        super().__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.encoder = Encoder(seq_len, n_features, embedding_dim)
        self.decoder_1 = Decoder(seq_len, embedding_dim,  n_features)
        self.decoder_2 = Decoder(seq_len, embedding_dim, n_features)
        self.learning_rate = learning_rate
        self.mse = torch.nn.MSELoss()
        self.mse_per_batch = torch.nn.MSELoss(reduction='none')

    def forward(self, x, alpha=0.5, beta=0.5):
        w1 = self.decoder_1(self.encoder(x)).reshape((-1, self.seq_len * self.n_features))
        w2 = self.decoder_2(self.encoder(w1)).reshape((-1, self.seq_len * self.n_features))
        x = x.reshape((-1, self.seq_len * self.n_features))

        loss = alpha * self.mse_per_batch(x, w1) + \
               beta * self.mse_per_batch(x, w2)

        return torch.mean(loss, dim=1)

    def configure_optimizers(self):
        optimizer_1 = Adam(list(self.encoder.parameters()) + list(
            self.decoder_1.parameters()), lr=self.learning_rate)
        optimizer_2 = Adam(list(self.encoder.parameters()) + list(
            self.decoder_2.parameters()), lr=self.learning_rate)

        return optimizer_1, optimizer_2

    def training_step(self, train_batch, batch_idx, optimizer_idx):
        n = self.trainer.current_epoch + 1

        z = self.encoder(train_batch)
        w1 = self.decoder_1(z)
        w22 = self.decoder_2(self.encoder(w1))

        # Train AE1
        if optimizer_idx == 0:
            loss1 = self.mse(train_batch, w1) + \
                    self.mse(train_batch, w22)
            output = OrderedDict({"loss": loss1})
            return output

        if optimizer_idx == 1:
            w2 = self.decoder_2(z)
            loss2 = self.mse(train_batch, w2) + \
                    self.mse(train_batch, w22)
            output = OrderedDict({"loss": loss2})
            return output

    def validation_step(self, test_batch, batch_idx):
        n = self.trainer.current_epoch + 1
        z = self.encoder(test_batch)
        w1 = self.decoder_1(z)

        w22 = self.decoder_2(self.encoder(w1))

        w2 = self.decoder_2(z)

        loss2 = 1 / n * torch.mean((test_batch - w2) ** 2) - \
                (1 - 1 / n) * torch.mean((test_batch - w22) ** 2)
        output = OrderedDict({"val_loss": loss2})
        self.logger.log_metrics({'val_loss': loss2.item()})
        self.log('val_loss', loss2, prog_bar=True)
        return output

    def validation_epoch_end(self, validation_step_outputs):
        temp = []
        for output in validation_step_outputs:
            temp += [output["val_loss"].item()]
        loss = torch.mean(torch.tensor(temp))
        self.logger.log_metrics({'val_loss': loss.item()})
        return {"val_loss": loss}