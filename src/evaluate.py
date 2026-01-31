import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from preprocess import load_data
from model import SpineMulticlassClassifier


def evaluate_model(model, X_test, y_test):

    model.eval()

    with torch.inference_mode():
        logits = model(X_test)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

    y_true = y_test.numpy()
    y_pred = preds.numpy()

    print("\n📊 Evaluation Metrics\n")

    print("Accuracy :", round(accuracy_score(y_true, y_pred), 4))
    print("Precision:", round(precision_score(y_true, y_pred, average='macro'), 4))
    print("Recall   :", round(recall_score(y_true, y_pred, average='macro'), 4))
    print("F1-score :", round(f1_score(y_true, y_pred, average='macro'), 4))

    print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))

    print("\nClassification Report:\n")
    print(classification_report(
        y_true,
        y_pred,
        target_names=["Normal", "Hernia", "Spondylolisthesis"]
    ))


if __name__ == "__main__":

    X_train, X_test, y_train, y_test = load_data("../data/column_3C_weka.csv")

    model = SpineMulticlassClassifier()
    model.load_state_dict(torch.load("../artifacts/model.pth"))

    evaluate_model(model, X_test, y_test)
