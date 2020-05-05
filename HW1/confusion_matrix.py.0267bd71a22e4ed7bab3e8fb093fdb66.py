from sklearn.metrics import confusion_matrix
with open("features.json", mode="r") as stream:
    data = json.load(stream)