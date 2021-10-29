import os
import json
import pandas as pd
import numpy as np
from typing import List

dataset_dir = 'data/'
target_dir = 'data/trainingSet'
QuAD_path_train = 'train-v2.0.json'
QuAD_path_dev = 'dev-v2.0.json'
CoQA_path_train = 'coqa-train-v1.0.json'
CoQA_path_dev = 'coqa-dev-v1.0.json'


def read_quad(filename: str) -> pd.DataFrame:
    path = os.path.join(dataset_dir, filename)
    source = []
    target = []
    with open(path, 'r') as dataset:
        data = json.load(dataset)
        for topic in data['data']:
            for paragraph in topic['paragraphs']:
                context = paragraph['context']
                for q in paragraph['qas']:
                    if not q['is_impossible']:
                        question = q['question']
                        answer = q['answers'][0]['text']
                        target.append(question)
                        source.append(concat_line(answer, context))

    return create_df(source, target)


def read_coqa(filename: str) -> pd.DataFrame:
    path = os.path.join(dataset_dir, filename)
    source = []
    target = []
    with open(path, 'r') as dataset:
        data = json.load(dataset)
        for topic in data['data']:
            context = topic['story']
            for q in topic['questions']:
                question = q['input_text']
                target.append(question)
            for a in topic['answers']:
                answer = a['input_text']
                source.append(concat_line(answer, context))
    return create_df(source, target)


def write_csv(coqa_df: pd.DataFrame, quad_df: pd.DataFrame, filename: str):
    result = pd.concat([coqa_df, quad_df], axis=0).to_frame()
    output = os.path.join(target_dir, filename)
    result.to_csv(output, encoding="utf-8")  # noqa


def concat_line(answer_string: str, context_string: str) -> str:
    return f'answer_token <{answer_string}> context_token <{context_string}>'


def create_df(source: List[str], target: List[str]) -> pd.DataFrame:
    result = pd.DataFrame(columns=['text', 'question'])
    result['text'] = np.array(source)
    result['question'] = np.array(target)
    return result
