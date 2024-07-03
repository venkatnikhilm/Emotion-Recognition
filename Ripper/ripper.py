import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import wittgenstein as lw
from sklearn.metrics import precision_score, recall_score
import pickle
from tqdm import tqdm 

data = pd.read_csv("features_data_columns.csv").drop(["Unnamed: 0"], axis = 1)
print(len(data))
# data = data[data["Label"] != "Surprise"]
# data = data[data["Label"] != "Fear"]
# data = data[data["Label"] != "Disgust"]
# data.loc[data["Label"] == "Surprise", "Label"] = "Happy"
# data.loc[data["Label"] == "Fear", "Label"] = "Sad"
# data.loc[data["Label"] == "Disgust", "Label"] = "Angry"
# labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
labels = ["Neutral", "Happy", "Angry", "Sad", "Surprise", "Fear", "Disgust"]
# labels = ["Neutral", "Happy", "Angry", "Sad"]
# labels = ["Angry", "Happy", "Neutral", "Sad"]
# drop_labels = ["Disgust", "Fear", "Sad", "Surprise"]

# data = data[data["Label"] != "Disgust"]
# data = data[data["Label"] != "Fear"]
# # data = data[data["Label"] != "Sad"]
# data = data[data["Label"] != "Surprise"]

def get_label_data(label, data):
    label_data = data
    label_data["Emotion"] = label_data["Label"].apply(lambda x: 1 if x == label else 0)
    label_data = label_data.drop("Label", axis = 1)
    return label_data

def ripper(label, train, test):
    label_data_train = get_label_data(label, train)
    X_train = label_data_train.iloc[:, :-1]
    y_train = label_data_train.iloc[:, -1]
    
    label_data_test = get_label_data(label, test)
    X_test = label_data_test.iloc[:, :-1]
    y_test = label_data_test.iloc[:, -1]

    print(label + "\n")
    ripper_clf = lw.RIPPER()
    ripper_clf.fit(X_train, y_train, class_feat = "Emotion", pos_class = 1)
    # results["Image"] = X_test.index
    results[label] = ripper_clf.predict(X_test)
    accuracy = ripper_clf.score(X_test, y_test)
    precision = ripper_clf.score(X_test, y_test, precision_score)
    recall = ripper_clf.score(X_test, y_test, recall_score)
    print(f'Accuracy: {accuracy} Precision: {precision} Recall: {recall}')
    print("-" * 50)
    return ripper_clf
    
def get_features(image, threshold):
    out = []
    for column in data.columns[:-1]:
        if float(data.loc[image][column]) >= threshold:
            out.append(column)
    return out

def get_check_labels(row):
    out = []
    for label in labels:
        if row[label]:
            out.append(label)
    return out

def get_cond_tuple(cond):
    feature_name = cond.feature
    cond_type = 0 if cond.val[0] == "<" else 1 if " - " in cond.val else 2
    if cond_type == 0 or cond_type == 2:
        value = float(cond.val[1:])
        return feature_name, cond_type, value
    value1, value2 = map(lambda x: float(x.strip()), cond.val.split(" - "))
    return feature_name, cond_type, value1, value2
        
def ripper_fuzzy():
    results["Count"] = results[labels].sum(axis = 1)
    results["Final Label"] = "No Emotion"
    for i, row in results[results["Count"] == 1].iterrows():
        for label in labels:
            if row[label]:
                results.loc[results["Image"] == row["Image"], "Final Label"] = label
                break
    
    print("1 Emo", len(results[results["Final Label"] != "No Emotion"]))
    
    new_data = results[results["Count"] != 1]
    for i, row in new_data.iterrows():
        high_prob = 0
        best_label = "None"  
        image = row["Image"]
        check_labels = labels[::-1] if row["Count"] == 0 else get_check_labels(row)
        for label in check_labels:
            covered = 0
            tot = 0
            ripper_clf = ripper_dict[label]
            for rule in ripper_clf.ruleset_:
                conds = [get_cond_tuple(item) for item in rule.conds]
                for cond in conds:
                    if cond[1] == 0 and data.loc[image][cond[0]] <= cond[2]:
                        covered += 1
                    elif cond[1] == 1 and data.loc[image][cond[0]] >= cond[2] and data.loc[image][cond[0]] <= cond[3]:
                        covered += 1
                    elif data.loc[image][cond[0]] >= cond[2]:
                        covered += 1
                tot += len(conds)
            coverage = covered / tot
            if coverage > high_prob:
                high_prob = coverage
                best_label = label
        results.loc[results["Image"] == image, "Final Label"] = best_label
        
    print("No Emo", len(results[results["Final Label"] == "No Emotion"]))
    print("None", len(results[results["Final Label"] == "None"]))
    print("Tot", len(results))
    print("Fuzzy", len(results[results["Final Label"] != "None"]))
        
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size = 0.2, random_state = 42)
train_dataset = pd.concat([X_train, y_train], axis = 1)
test_dataset = pd.concat([X_test, y_test], axis = 1)

print(test_dataset)

ripper_dict = {}
results = pd.DataFrame()
results["Image"] = test_dataset.index
# results["Label"] = y_test
for label in labels:
    ripper_dict[label] = ripper(label, train_dataset, test_dataset)
with open("Ripper/Split/ripper_labels_objects_no_scale_main_4_emotions.prsg", "wb") as file:
    pickle.dump(ripper_dict, file)

results.to_csv("Ripper/Split/ripper_results_no_scale_main_4_emotions.csv")

# results = pd.read_csv("Ripper/Split/ripper_results_no_scale_all_emotions.csv").drop(["Unnamed: 0"], axis = 1)
# with open("Ripper/Split/ripper_labels_objects_no_scale_all_emotions.prsg", "rb") as file:
#     ripper_dict = pickle.load(file)
ripper_fuzzy()

results["Actual Label"] = results["Image"].apply(lambda x: test_dataset.loc[x]["Label"])
print(results)

for label in labels:
    print(label)
    mini = results[results["Actual Label"] == label]
    c, ic = 0, 0
    for i, row in mini.iterrows():
        if row["Actual Label"] == row["Final Label"]:
            c += 1
        else:
            ic += 1
    print("Correct:", c)
    print("Incorrect:", ic)
    print("Acc:", c / (c + ic))