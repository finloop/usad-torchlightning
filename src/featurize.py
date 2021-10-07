import os
import sys

import numpy as np
import pandas as pd
from sklearn import preprocessing
import yaml


def create_windows(data: np.ndarray, window_size):
    return data[np.arange(window_size) + np.arange(
        data.shape[0] - window_size).reshape(-1, 1)]


def load_dataset(filename, nrows, sep, decimal):
    df = pd.read_csv(filename, nrows=nrows, decimal=decimal, sep=sep, low_memory=False)
    labels_ = np.array([float(label != 'Normal') for label in
                        df["Normal/Attack"].values])
    df = df.drop(["Timestamp", "Normal/Attack"], axis=1)

    for i in list(df):
        df[i] = df[i].apply(lambda x: str(x).replace(",", "."))

    return df.astype(float), labels_


if __name__ == "__main__":
    # Read YAML params
    params = yaml.safe_load(open('params.yaml'))['featurize']
    max_row_limit = params["max_row_limit"]
    window_size = params["window_size"]

    # Read command line params
    if len(sys.argv) != 3:
        sys.stderr.write('Arguments error. Usage:\n')
        sys.stderr.write(
            '\tpython featurize.py data-dir-path features-dir-path\n'
        )
        sys.exit(1)

    data_dir = sys.argv[1]
    out_dir = sys.argv[2]

    os.makedirs(out_dir, exist_ok=True)

    normal_csv = os.path.join(data_dir, "SWaT_Dataset_Normal_v1.csv")
    attack_csv = os.path.join(data_dir, "SWaT_Dataset_Attack_v0.csv")

    train_file = os.path.join(out_dir, "data.npz")

    normal, _ = load_dataset(normal_csv, nrows=max_row_limit, sep=",", decimal=",")
    attack, labels = load_dataset(attack_csv, nrows=max_row_limit, sep=";", decimal=";")

    sc = preprocessing.StandardScaler()

    normal = sc.fit_transform(normal.values)
    attack = sc.transform(attack.values)

    windows_normal = create_windows(normal, window_size).reshape(-1, normal.shape[1]*window_size)
    windows_attack = create_windows(attack, window_size).reshape(-1, attack.shape[1]*window_size,)

    np.savez_compressed(train_file, train=windows_normal, test=windows_attack, labels=labels)
