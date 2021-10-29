import os
from readers.DatasetReader import *

QuAD_path_train = 'train-v2.0.json'
QuAD_path_dev = 'dev-v2.0.json'
CoQA_path_train = 'coqa-train-v1.0.json'
CoQA_path_dev = 'coqa-dev-v1.0.json'


def run():
    coqa_train = read_coqa(CoQA_path_train)
    quad_train = read_quad(QuAD_path_train)
    write_csv(coqa_train, quad_train, 'train.csv')


if __name__ == '__main__':
    run()
