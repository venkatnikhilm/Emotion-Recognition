import pickle
import wittgenstein as lw
with open("Ripper/Split/ripper_labels_objects_no_scale_all_emotions.prsg", "rb") as file:
    d = pickle.load(file)

labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
for label in labels:
    d[label].out_model()
    print()
    for rule in d[label].ruleset_:
        conds = [(item.feature, item.val) for item in rule.conds]
        print(conds)
    break