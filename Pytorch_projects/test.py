import pandas as pd

# Load the dataset
# Load dataset
df = pd.read_csv("data.csv").dropna()

X = df.drop(columns=['t', 'time'], axis=1).values
y = df['t'].values

print(Y)
