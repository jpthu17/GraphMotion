import os
import argparse
import json

from allennlp.predictors.predictor import Predictor


def save_json(save_path, data):
    with open(save_path, "w") as file:
        json.dump(data, file)


def load_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def process(graph):
    V, entities, relations = {}, {}, []
    for i in graph["verbs"]:
        description = i['description']
        pos = 0
        flag = 0
        _words, _spans = [], []
        tags, verb = {}, 0
        for i in description.split():
            if "[" in i:
                _role = i[1:-1]
                flag = 1
                _spans = [pos]
                _words = []

            elif "]" in i:
                _words.append(i[:-1])
                pos += 1
                flag = 0
                if _role == "V":
                    V[len(V)] = {
                        "role": _role,
                        "spans": _spans,
                        "words": _words
                    }
                    verb = len(V) - 1
                else:
                    entities[len(entities)] = {
                        "role": _role,
                        "spans": _spans,
                        "words": _words
                    }
                    tags[len(entities) - 1] = _role
            else:
                pos += 1
                if flag:
                    _words.append(i)
                    _spans.append(pos)

        for i in tags:
            relations.append((verb, i, tags[i]))

    output = {
        "V": V,
        "entities": entities,
        "relations": relations
    }
    return output


if __name__ == '__main__':
    predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
    print("test data")
    test_data = load_json("test_data.json")
    new_test_data = test_data
    for key in test_data:
        for id in range(len(test_data[key])):
            temp = test_data[key][id]
            graph = process(predictor.predict_tokenized(temp["caption"].split()))
            temp["V"] = graph["V"]
            temp["entities"] = graph["entities"]
            temp["relations"] = graph["relations"]
            new_test_data[key][id] = temp

    save_json("new_test_data.json", new_test_data)

    print("train data")
    train_data = load_json("train_data.json")
    new_train_data = train_data
    for key in train_data:
        for id in range(len(train_data[key])):
            temp = train_data[key][id]
            graph = process(predictor.predict_tokenized(temp["caption"].split()))
            temp["V"] = graph["V"]
            temp["entities"] = graph["entities"]
            temp["relations"] = graph["relations"]
            new_train_data[key][id] = temp

    save_json("new_train_data.json", new_train_data)

if __name__ == '__main__1':
    predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
    sent = "a man walks forward, then squats to pick something up with both hands, stands back up, and resumes walking."
    out = predictor.predict_tokenized(sent.split())
    print(out)
    print(process(out))