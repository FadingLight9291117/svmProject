import json
import csv
import pandas as pd

import paddlehub as hub
from data.dataset import dataset

senta = hub.Module(name="senta_lstm")

test_text = dataset()

results = senta.sentiment_classify(data={"text": test_text})

results = [
    {'text': item['text'], 'sentiment': item['sentiment_key']} for item in results
]

print("data length is {}".format(len(results)))

with open('./data/data.json', 'w+', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False)

with open('./data/data.csv', 'w+', encoding='utf-8') as f:
    writer = csv.DictWriter(f, results[0].keys())
    writer.writeheader()
    writer.writerows(results)

with open('./data/data.txt', 'w+', encoding='utf-8') as f:
    f.writelines(map(lambda item: '{}\t{}\n'.format(*item.values()), results))
