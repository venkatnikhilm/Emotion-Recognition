from collections import defaultdict, Counter
from itertools import combinations
import random

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori as a2
import pandas as pd

APRIORI_SUPPORT_THRESHOLD = 0.3
APRIORI_CONFIDENCE_THRESHOLD = 1
SUPPORT_THRESHOLD = 0.6

labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

facial_features = ['_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawOpen', 'jawRight', 'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight']
def get_feature_list(feature_list):
    feature_list = set(feature_list)
    out = []
    for feature in facial_features:
        if feature in feature_list:
            out.append(feature)
        else:
            out.append(f"!{feature}")
    return out

def apriori(data, min_support, min_confidence, give_all_rules = False):
    rules = {}

    # Get Inital Items
    init = []
    for i in data:
        for q in i:
            if q not in init:
                init.append(q)
    init = sorted(init)

    # Get Frequent Inits.
    c = Counter()
    for i in init:
        for d in data:
            if i in d:
                c[i] += 1

    s = int(min_support * len(init))
    l = Counter()
    for i in c:
        if c[i] >= s:
            l[frozenset([i])] += c[i]

    pl = l
    pos = 1
    for count in range(2,1000):
        nc = set()
        temp = list(l)
        for i in range(0, len(temp)):
            for j in range(i + 1, len(temp)):
                t = temp[i].union(temp[j])
                if len(t) == count:
                    nc.add(temp[i].union(temp[j]))
        nc = list(nc)

        c = Counter()
        for i in nc:
            c[i] = 0
            for q in data:
                temp = set(q)
                if i.issubset(temp):
                    c[i] += 1

        l = Counter()
        for i in c:
            if c[i] >= s:
                l[i] += c[i]

        if len(l) == 0:
            break
        if give_all_rules:
            rules.update(get_rules(get_association_rules(l, data), min_confidence))
        pl = l
        pos = count

    for l in pl:
        c = [frozenset(q) for q in combinations(l,len(l)-1)]
        mmax = 0
        for a in c:
            b = l - a
            ab = l
            sab = 0
            sa = 0
            sb = 0
            for q in data:
                temp = set(q)
                if a.issubset(temp):
                    sa += 1
                if b.issubset(temp):
                    sb += 1
                if ab.issubset(temp):
                    sab += 1
            temp = sab / sa * 100
            if temp > mmax:
                mmax = temp
            temp = sab / sb * 100
            if temp > mmax:
                mmax = temp

        curr = 1
        for a in c:
            b = l - a
            ab = l
            sab = 0
            sa = 0
            sb = 0
            for q in data:
                temp = set(q)
                if(a.issubset(temp)):
                    sa+=1
                if(b.issubset(temp)):
                    sb+=1
                if(ab.issubset(temp)):
                    sab+=1
            temp = sab/sa*100
            curr += 1
            temp = sab/sb*100
            curr += 1

    itemset = []
    for key in pl.keys():
        itemset.append(key)
    if not give_all_rules:
        rules = get_rules(get_association_rules(pl, data), min_confidence)
    return itemset, rules

def get_association_rules(l_set, data):
    out = ""
    for l in l_set:
        c = [frozenset(q) for q in combinations(l,len(l)-1)]
        mmax = 0
        for a in c:
            b = l-a
            ab = l
            sab = 0
            sa = 0
            sb = 0
            for q in data:
                temp = set(q)
                if(a.issubset(temp)):
                    sa+=1
                if(b.issubset(temp)):
                    sb+=1
                if(ab.issubset(temp)):
                    sab+=1
            temp = sab/sa*100
            if(temp > mmax):
                mmax = temp
            temp = sab/sb*100
            if(temp > mmax):
                mmax = temp
            out += str(list(a))+" -> "+str(list(b))+" = "+str(sab/sa*100)+"%\n"
            out += str(list(b))+" -> "+str(list(a))+" = "+str(sab/sb*100)+"%\n"
    return out

def breakdown_rule(rule, threshold):
    rule.strip()
    if rule == "":
        return None, None
    left, right = rule.split("->")
    right, confidence = right.split("=")
    confidence = float(confidence.strip()[:-1])
    if confidence < (threshold * 100):
        return None, None
    left = left.strip()[1:-1]
    left = [x.strip()[1:-1] for x in left.split(",")]
    right = right.strip()[1:-1]
    right = [x.strip()[1:-1] for x in right.split(",")]
    return left, right
def get_rules(text, confidence):
    rules = defaultdict(set)
    rules_text = text.split("\n")
    for rule in rules_text:
        left, right = breakdown_rule(rule, confidence)
        if left is None or right is None:
            continue
        rules[frozenset(left)].update(right)
    return rules

def clean_output(itemset, rules):
    for i, item in enumerate(itemset):
        itemset[i] = set(item)
    rule_list = []
    for k, v in rules.items():
        rule_list.append((set(k), set(v)))
    rule_list.sort(key = lambda x: (len(x[0]), len(x[1])))
    return itemset, rule_list

# Get Data
data = defaultdict(dict)
with open(f"features_data_{SUPPORT_THRESHOLD}.csv") as file:
    for line in file:
        values = list(map(lambda x: x.strip(), line.split(",")))
        data[values[1]][values[0]] = values[2:]

class Emotion:
    def __init__(self, itemset, rules, label):
        self.itemset = itemset
        self.rules = rules
        self.label = label

emotion_outputs = {}
train = {}
test = {}
a2_output = {}
for label in labels:
    train_keys = random.sample([*data[label].keys()], int(len(data[label]) * 0.7))
    test_keys = list(data[label].keys() - set(train_keys))
    train[label] = train_keys
    test[label] = test_keys

    # features = [get_feature_list(data[label][image]) for image in train[label]]
    features = [data[label][image] for image in train[label]]
    itemset, rules = apriori(features, APRIORI_SUPPORT_THRESHOLD, APRIORI_CONFIDENCE_THRESHOLD, True)
    itemset, rules = clean_output(itemset, rules)
    emotion_outputs[label] = Emotion(itemset, rules, label)
    print(f"{label} Done")
    # break
    
# for label in labels:
#     print(label)
#     for item in emotion_outputs[label].itemset:
#         print(item)
#     print()
#     for rule in emotion_outputs[label].rules:
#         print(rule)
#     print("-"*50)

# for label in labels:
#     i = 0
#     while i < len(emotion_outputs[label].rules):
#         if label in emotion_outputs[label].rules[i][0]:
#             emotion_outputs[label].rules.pop(i)
#         else:
#             i += 1
# print("-"*50)

for label in labels:
    for image in test[label][:10]:
        emotions = []
        for test_label in labels:
            features = set(data[label][image])
            for key, value in emotion_outputs[test_label].rules:
                if key.issubset(features):
                    features.update(set(value))
            for item in emotion_outputs[test_label].itemset:
                if item.issubset(features):
                    emotions.append(test_label)
                    break
        print(label, emotions, label in emotions, len(emotions) - (1 if label in emotions else 0))

# for label in ["Angry"]:
#     c, ic = 0, 0
#     for image in test[label][:10]:
#         features = set(data[label][image])
#         for key, value in emotion_outputs[label].rules:
#             if key.issubset(features):
#                 features.update(set(value))
#         for item in emotion_outputs[label].itemset:
#             print(item)
#             print(features)
#             if item.issubset(features):
#                 print("MATCH")
#                 c += 1
#                 break
#         else:
#             print("F")
#             ic += 1
#     print(label, c, ic)
# out = defaultdict(dict)
# for label in labels:
#     for test_label in labels:
#         c, ic = 0, 0
#         for image in test[test_label]:
#             features = set(data[test_label][image])
#             ol = len(features)
#             for key, value in emotion_outputs[label].rules:
#                 if key.issubset(features):
#                     features.update(set(value))
#             el = len(features)
#             if ol != el:
#                 print(image, ol, el)
#             for item in emotion_outputs[label].itemset:
#                 if set(item).issubset(features):
#                     c += 1
#                     break
#             else:
#                 ic += 1
#         out[label][test_label] = (c, ic) if label == test_label else (ic, c)
# for a in labels:
#     for b in labels:
#         print(f"{a}-{b} -> Correct: {out[a][b][0]}, Incorrect: {out[a][b][1]}")