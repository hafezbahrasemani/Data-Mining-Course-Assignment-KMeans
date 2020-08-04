import numpy as np


def read_from_file(path_to_file):
    with open(path_to_file, newline='') as file:
        X = []
        Y = []
        next(file)
        for line in file:
            tokens = line.split(',')
            X.append(float(tokens[0]))
            Y.append(float(tokens[1]))

    return X, Y