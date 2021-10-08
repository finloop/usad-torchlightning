import yaml
import sys
import os
import matplotlib.pylab as plt
import numpy as np
import json
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import roc_curve  # Calculate the ROC curve
from sklearn.metrics import precision_recall_curve  # Calculate the Precision-Recall curve
from sklearn.metrics import f1_score, recall_score, accuracy_score

if __name__ == "__main__":
    params = yaml.safe_load(open('params.yaml'))

    if len(sys.argv) != 4:
        sys.stderr.write('Arguments error. Usage:\n')
        sys.stderr.write(
            '\tpython validate.py featurize-dir-path predict-dir-path metrics-file\n'
        )
        sys.exit(1)

    featurize_dir = sys.argv[1]
    predict_dir = sys.argv[2]
    metrics_file = sys.argv[3]

    y_pred = np.load(os.path.join(predict_dir, "y_pred.npz"))["y_pred"]
    labels = np.load(os.path.join(featurize_dir, "data.npz"))["labels"]

    WINDOW_SIZE = params["featurize"]["window_size"]
    windows_labels = []
    for i in range(len(labels) - WINDOW_SIZE):
        windows_labels.append(list(np.int_(labels[i:i + WINDOW_SIZE])))

    y_test = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels]


    thresholds = np.arange(0.0, np.max(y_pred), np.max(y_pred)/50)
    fscore = np.zeros(shape=(len(thresholds)))
    #rscore = np.zeros(shape=(len(thresholds)))

    # Fit the model
    for index, elem in enumerate(thresholds):
        # Corrected probabilities
        y_pred_prob = (y_pred > elem).astype('int')
        # Calculate the f-score
        fscore[index] = f1_score(y_test, y_pred_prob)
        #rscore[index] = recall_score(y_test, y_pred_prob)

    index = np.argmax(fscore)
    thresholdOpt = round(thresholds[index], ndigits=4)
    fscoreOpt = round(fscore[index], ndigits=4)
    y_pred_prob = (y_pred > thresholds[index]).astype('int')
    acc = accuracy_score(y_test, y_pred_prob)
    recall = recall_score(y_test, y_pred_prob)

    # save scores
    with open(metrics_file, 'w') as f:
        json.dump({'threshold': thresholdOpt, "acc": acc, "recall": recall, "f1": fscoreOpt}, f)
