import numpy as np
import yaml
from model import USADModel
from torch.utils.data import DataLoader, Dataset
import sys
import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

# Create Dataset
class NpzDataset(Dataset):
    def __init__(self, path, key="data"):
        self.path = path
        self.data = np.load(path)[key]

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]).float()

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    # Read YAML params
    params = yaml.safe_load(open('params.yaml'))

    WINDOW_SIZE = params['featurize']['window_size']
    BATCH_SIZE = params['train']["batch_size"]
    EPOCHS = params['train']["epochs"]
    HIDDEN_SIZE = params['train']["hidden_size"]
    S1 = int(params['train']["s1"])
    S2 = int(params['train']["s2"])

    if len(sys.argv) != 3:
        sys.stderr.write('Arguments error. Usage:\n')
        sys.stderr.write(
            '\tpython featurize.py features-dir-path predict-dir-path\n'
        )
        sys.exit(1)

    data_dir = sys.argv[1]
    predict_dir = sys.argv[2]

    os.makedirs(predict_dir, exist_ok=True)

    data_file = os.path.join(data_dir, "data.npz")

    test = NpzDataset(data_file, "test")
    train = NpzDataset(data_file, "train")

    test_loader = DataLoader(test, batch_size=BATCH_SIZE, num_workers=5)
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, num_workers=5)

    NMETRICS = test[0].size()[0] // WINDOW_SIZE

    model = USADModel(window_size=WINDOW_SIZE * NMETRICS, z_size=WINDOW_SIZE * HIDDEN_SIZE, s1=S1, s2=S2)

    checkpoint_callback = ModelCheckpoint(
        monitor="loss",
        dirpath=predict_dir,
        filename="usad-{epoch:02d}-{loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    trainer = Trainer(gpus=1, max_epochs=EPOCHS, callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader, test_loader)

    y_pred = trainer.predict(model, test_loader)
    y_pred = np.concatenate([torch.stack(y_pred[:-1]).flatten().detach().cpu().numpy(),
                             y_pred[-1].flatten().detach().cpu().numpy()])

    np.savez_compressed(os.path.join(predict_dir, "y_pred.npz"), y_pred=y_pred)