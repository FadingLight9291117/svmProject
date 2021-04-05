import json
from pathlib import Path

data_path = Path('data/data.json')

data = json.load(data_path.open('r', encoding='utf-8'))

TOTAL_NUM = len(data)

POSITIVE_NUM = len([v for v in data if v['sentiment'] == 'positive'])

NEGATIVE_NUM = len([v for v in data if v['sentiment'] == 'negative'])
