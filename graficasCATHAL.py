import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#leo el csv
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data', names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"])
#nombre de las columnas
print(df.columns)
#cambio los caracteres no numericos
df['ca'].replace('?', np.nan, inplace=True)
df["ca"] = df["ca"].astype(float)
df['thal'].replace('?', np.nan, inplace=True)
df["thal"] = df["thal"].astype(float)
print(df['ca'].dtype)
df.head()
#creo un nuevo dataframe con 2 columnas
aja =["ca", "thal"]
graficas =df[aja]
#histograma de frecuencias
graficas.hist(figsize=(10,10))
plt.show()