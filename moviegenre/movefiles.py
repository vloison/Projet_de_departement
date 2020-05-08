import pathlib
import pandas as pd

source = pathlib.Path('../data2/posters')
df = pd.read_csv('movefiles.csv')

for cat in movefiles.category.unique():
    p = source/cat
    p.mkdir()
    for genre in movefiles.genre.unique():
        q = p/genre
        q.mkdir()

p = source/'autres'
p.mkdir()

for i, row in df.iterrows():
    s = source/row['name']
    if s.exists():
        s.replace(source/row['category']/row['genre']/row['name'])

for autre in source.glob('*.jpg'):
    autre.replace(source/'autres'/autre.name)



