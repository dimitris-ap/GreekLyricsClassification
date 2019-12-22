import pandas as pd
import numpy as np

df = pd.read_csv('dataset/greekLyrics_ansi.csv', delimiter=';', encoding='ANSI')
print(df.info())

print("All genres:", df['genre'].unique())
