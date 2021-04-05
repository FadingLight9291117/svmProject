from pathlib import Path
import pandas as pd


def dataset():
    df = pd.read_excel('data/comments.xls')
    data = df[df.keys()[0]]
    data = [item for item in data.to_list() if str(item) != 'nan']
    return data


if __name__ == '__main__':
    data = dataset()
