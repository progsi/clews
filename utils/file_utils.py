import sys
import csv
import json


def load_txt(fn):
    with open(fn, "r") as fh:
        lines = fh.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].replace("\n", "")
        lines[i] = lines[i].replace("\r", "")
    return lines


def load_csv(fn, sep=",", header=0, quotechar='"'):
    lines = load_txt(fn)
    csv_reader = csv.reader(
        lines,
        quotechar=quotechar,
        quoting=csv.QUOTE_ALL,
        delimiter=sep,
    )
    data = []
    for i, l in enumerate(csv_reader):
        if i == 0:
            desc = l[:]
            for _ in l:
                data.append([])
        elif len(l) != len(data):
            print("Error reading " + fn)
            sys.exit()
        if i < header:
            continue
        for j, item in enumerate(l):
            data[j].append(item)
    return desc, data, len(data[0])


def load_json(fn):
    with open(fn, "r") as fh:
        d = json.load(fh)
    return d


def load_jsons(fn, limit_lines=None):
    with open(fn, "r") as fh:
        d = []
        for line in fh:
            aux = json.loads(line)
            d.append(aux)
            if limit_lines is not None and len(d) == limit_lines:
                break
    return d
