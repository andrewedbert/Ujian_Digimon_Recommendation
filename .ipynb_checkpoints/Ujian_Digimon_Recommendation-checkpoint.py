import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_json('digimon.json')
df = df[['no','digimon','stage','type','attribute']]
# print(df)

model = CountVectorizer(
    tokenizer = lambda i: i.split('😎'),
    analyzer = 'word',
)

matrix = model.fit_transform(df['digimon'])
digimon = model.get_feature_names()
countdigi = len(digimon)
dfmx = matrix.toarray()

print(digimon)
print(countdigi)
print(dfmx)
