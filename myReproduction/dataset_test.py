import pandas as pd

dir = "../../Data/brennan2019/S01/events.csv"
events = pd.read_csv(dir)
word_mask = events['kind'] == 'word'
words = events['word'].values[word_mask]
word_set = set(words)
print(len(words))
print(len(word_set))