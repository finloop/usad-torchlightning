import numpy as np
import yaml
from model import USADModel
from torch.utils.data import DataLoader, Dataset
import sys
import os
import torch
from pytorch_lightning import Trainer

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

    test_loader = DataLoader(test, batch_size=BATCH_SIZE, num_workers=3)
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, num_workers=3)

    NMETRICS = test[0].size()[0] // WINDOW_SIZE

    model = USADModel(window_size=WINDOW_SIZE * NMETRICS, z_size=WINDOW_SIZE * HIDDEN_SIZE)

    trainer = Trainer(gpus=1, max_epochs=EPOCHS, )

    trainer.fit(model, train_loader, train_loader)

    y_pred = trainer.predict(model, test_loader)

    y_pred = np.concatenate([torch.stack(y_pred[:-1]).flatten().detach().cpu().numpy(),
                             y_pred[-1].flatten().detach().cpu().numpy()])

    np.savez_compressed(os.path.join(predict_dir, "y_pred.npz"), y_pred=y_pred)