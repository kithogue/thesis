import os
import json
import pandas as pd

dataset_dir = 'data/'
target_dir = 'data/trainingSet'
QuAD_path_train = 'train-v2.0.json'
QuAD_path_dev = 'dev-v2.0.json'
CoQA_path_train = 'coqa-train-v1.0.json'
CoQA_path_dev = 'coqa-dev-v1.0.json'


def read_quad(filename: str) -> pd.DataFrame:
    pass


def read_coqa(filename: str) -> pd.DataFrame:
    pass


def write_csv(coqa_df: pd.DataFrame, quad_df: pd.DataFrame, filename: str):
    result = pd.concat([coqa_df, quad_df], axis=0).to_frame()
    output = os.path.join(target_dir, filename)
    result.to_csv(output, encoding="utf-8")


def concat_line(answer_string: str, context_string: str) -> str:
    return 'answer_token + <' + answer_string + '> context_token <' + context_string + '>'
