import pickle
import pandas as pd
from sklearn.metrics import precision_score, recall_score

with open("Ripper/Split/ripper_labels_objects_no_scale_all_emotions.prsg", "rb") as file:
    d = pickle.load(file)

data = pd.read_csv("features_data_columns.csv").drop(["Unnamed: 0"], axis = 1)
labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

for label in labels:
    mini = data[data["Label"] == label]
    mini = mini.assign(Emotion=1)
    mini = mini.drop("Label", axis = 1)
    ripper_clf = d[label]
    X_test = mini.iloc[:, :-1]
    y_test = mini.iloc[:, -1]
    print(label)
    accuracy = ripper_clf.score(X_test, y_test)
    precision = ripper_clf.score(X_test, y_test, precision_score)
    recall = ripper_clf.score(X_test, y_test, recall_score)
    print(f'Accuracy: {accuracy} Precision: {precision} Recall: {recall}')
    print("-" * 50)