import os
import tempfile
import click

import mlflow
import mlflow.sklearn

import numpy as np
import pandas as pd
from sklearn import preprocessing

@click.command(help="Generate windowed features from dataset.")
@click.option("--window-size", default=12, help="Set rolling window size")
@click.option("--max-row-limit", default=10000)
def featurize(window_size, max_row_limit):
    with mlflow.start_run() as mlrun:
        tmpdir = tempfile.mkdtemp()

        test_file = os.path.join(tmpdir,"train.npz")
        train_file = os.path.join(tmpdir,"test.npz")

        normal_csv = "data/SWaT_Dataset_Normal_v1.csv"
        attack_csv = "data/SWaT_Dataset_Attack_v0.csv"

        #
        # NORMAL DATA
        #
        normal = pd.read_csv(normal_csv, nrows=max_row_limit)
        normal = normal.drop(["Timestamp", "Normal/Attack"], axis=1)
        print(f"Loaded {len(normal)} train rows from {normal_csv}")

        # Transform all columns into float64
        for i in list(normal):
            normal[i] = normal[i].apply(lambda x: str(x).replace(",", "."))
        normal = normal.astype(float)

        sc = preprocessing.StandardScaler()
        x = normal.values
        x_scaled = sc.fit_transform(x)
        normal = pd.DataFrame(x_scaled)

        windows_normal = normal.values[np.arange(window_size)[None, :] + np.arange(
            normal.shape[0] - window_size)[:, None]]

        print(f"Created train windows of shape {windows_normal.shape}")
        print(f"Saving windows_normal to {train_file}")
        np.savez_compressed(train_file, train=train_file)
        mlflow.log_artifact(train_file, "train-file")
        print(f"Saved {train_file}")

        #
        # ATTACK DATA
        #
        attack = pd.read_csv(attack_csv,
                             sep=";", nrows=max_row_limit)
        labels = [float(label != 'Normal') for label in
                  attack["Normal/Attack"].values]
        attack = attack.drop(["Timestamp", "Normal/Attack"], axis=1)
        print(f"Loaded {len(attack)} test rows from {attack_csv}")

        # Transform all columns into float64
        for i in list(attack):
            attack[i] = attack[i].apply(lambda x: str(x).replace(",", "."))
        attack = attack.astype(float)

        attack = pd.DataFrame(sc.transform(attack.values))

        windows_attack = attack.values[np.arange(window_size)[None, :] + np.arange(
            attack.shape[0] - window_size)[:, None]]

        print(f"Created test windows of shape {windows_attack.shape}")

        print(f"Saving windows_attack to {test_file}")
        np.savez_compressed(test_file, test=test_file)
        print(f"Saved {test_file}")
        mlflow.log_artifact(test_file, "test-file")


if __name__ == "__main__":
    featurize()