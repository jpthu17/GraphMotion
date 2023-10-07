import json
import os


def save_json(save_path, data):
    with open(save_path, "w") as file:
        json.dump(data, file)


def load_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def process(graph):
    entities, relations = {}, []
    for i in graph["verbs"]:
        description = i['description']
        pos = 0
        flag = 0
        _words, _spans = [], []
        for i in description.split():
            tags, verb = {}, 0
            if "[" in i:
                _role = i[1:-1]
                flag = 1
                _spans = [pos]
                _words = []

            elif "]" in i:
                _words.append(i[:-1])
                entities[len(entities)] = {
                    "role": _role,
                    "spans": _spans,
                    "words": _words
                }
                pos += 1
                flag = 0
                if _role != "V":
                    tags[len(entities)] = _role
                else:
                    verb = len(entities)
            else:
                pos += 1
                if flag:
                    _words.append(i)
                    _spans.append(pos)

            for i in tags:
                relations.append((verb, i, tags[i]))

    output = {
        "entities": entities,
        "relations": relations
    }
    return output