from sklearn.metrics import confusion_matrix
import json

with open("predicts_test.json", mode="r") as stream:
    predicts = json.load(stream)

with open("golds_test.json", mode="r") as stream:
    golds = json.load(stream)

a = confusion_matrix(golds, predicts)
print(a)
