import torch
import torch.nn as nn
from torch.optim.adam import Adam
from pytorch_lightning.core.lightning import LightningModule


class Encoder(LightningModule):
    def __init__(self, input_size, latent_size):
        super().__init__()

        self.layer_1 = nn.Linear(input_size, input_size // 2)
        self.layer_2 = nn.Linear(input_size // 2, input_size // 4)
        self.layer_3 = nn.Linear(input_size // 4, latent_size)

        self.activation = nn.ReLU(True)

    def forward(self, x):
        out = self.layer_1(x)
        out = self.activation(out)
        out = self.layer_2(out)
        out = self.activation(out)
        out = self.layer_3(out)
        z = self.activation(out)
        return z


class Decoder(LightningModule):
    def __init__(self, latent_size, output_size):
        super().__init__()

        self.layer_3 = nn.Linear(latent_size, output_size // 4)
        self.layer_2 = nn.Linear(output_size // 4, output_size // 2)
        self.layer_1 = nn.Linear(output_size // 2, output_size)

        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.layer_1(x)
        out = self.activation(out)
        out = self.layer_2(out)
        out = self.activation(out)
        out = self.layer_3(out)
        w = self.sigmoid(out)
        return w


class USADModel(LightningModule):
    def __init__(self, window_size, z_size, learning_rate=1e-3):
        super().__init__()

        self.encoder = Encoder(window_size, z_size)
        self.decoder_1 = Decoder(z_size, window_size)
        self.decoder_2 = Decoder(z_size, window_size)
        self.learning_rate = learning_rate

    def forward(self, x, alpha=.5, beta=.5):
        w1 = self.decoder_1(x)
        w2 = self.decoder_2(w1)

        return alpha * torch.mean((x - w1)**2, axis=1) + \
               beta * torch.mean((x - w2)**2, axis=1)

    def configure_optimizers(self):
        optimizer_1 = Adam(list(self.encoder.parameters()) + list(
            self.decoder_1.parameters()), lr=self.learning_rate)
        optimizer_2 = Adam(list(self.encoder.parameters()) + list(
            self.decoder_2.parameters()), lr=self.learning_rate)

        return optimizer_1, optimizer_2

    def training_step(self, train_batch, batch_idx):
        n = batch_idx
        z = self.encoder(train_batch)
        w1 = self.decoder_1(z)
        w2 = self.decoder_2(z)
        w22 = self.decoder2(self.encoder(w1))
        loss1 = 1 / n * torch.mean((train_batch - w1) ** 2) + \
                (1 - 1 / n) * torch.mean((train_batch - w22) ** 2)
        loss2 = 1 / n * torch.mean((train_batch - w2) ** 2) - \
                (1 - 1 / n) * torch.mean((train_batch - w22) ** 2)

        self.log('train_loss_1', loss1)
        self.log('train_loss_2', loss1)

        return loss1, loss2

    def validation_step(self, val_batch, batch_idx):
        n = batch_idx
        z = self.encoder(val_batch)
        w1 = self.decoder_1(z)
        w2 = self.decoder_2(z)
        w22 = self.decoder2(self.encoder(w1))
        loss1 = 1 / n * torch.mean((val_batch - w1) ** 2) + \
                (1 - 1 / n) * torch.mean((val_batch - w22) ** 2)
        loss2 = 1 / n * torch.mean((val_batch - w2) ** 2) - \
                (1 - 1 / n) * torch.mean((val_batch - w22) ** 2)

        self.log('val_loss_1', loss1)
        self.log('val_loss_2', loss1)

        return loss1, loss2

