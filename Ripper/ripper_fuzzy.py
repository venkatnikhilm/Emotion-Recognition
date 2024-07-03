import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import wittgenstein as lw
from sklearn.metrics import precision_score, recall_score
import pickle

data = pd.read_csv("features_data_columns.csv").drop(["Unnamed: 0"], axis = 1)
labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
def get_label_data(label, data):
    label_data = data
    label_data["Emotion"] = label_data["Label"].apply(lambda x: 1 if x == label else 0)
    label_data = label_data.drop("Label", axis = 1)
    return label_data

def ripper(label, data = data):
    label_data = get_label_data(label, data)
    X = label_data.iloc[:, :-1]
    y = label_data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    print(label + "\n")
    ripper_clf = lw.RIPPER()
    ripper_clf.fit(X_train, y_train, class_feat = "Emotion", pos_class = 1)
    results[label] = ripper_clf.predict(X_test)
    accuracy = ripper_clf.score(X_test, y_test)
    precision = ripper_clf.score(X_test, y_test, precision_score)
    recall = ripper_clf.score(X_test, y_test, recall_score)
    print(f'Accuracy: {accuracy} Precision: {precision} Recall: {recall}')
    print("-" * 50)
    return ripper_clf

X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size = 0.2, random_state = 42)
train_dataset = pd.concat([X_train, y_train], axis = 1)
test_dataset = pd.concat([X_test, y_test], axis = 1)

print(train_dataset)