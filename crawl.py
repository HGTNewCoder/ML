import pandas as pd

a = pd.read_csv("gemma1000.csv")
b = pd.read_parquet("data.parquet", engine='pyarrow')

print(a)