import pandas as pd
import time
from sys import getsizeof

for chunk in pd.read_csv('../datasets/data/trainingRecords.csv', verbose=True, chunksize=10000):
    print(getsizeof(chunk))

for chunk in pd.read_csv('../datasets/data/validationRecords.csv', verbose=True, chunksize=10000):
    print(getsizeof(chunk))