import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Data.csv")

cols = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "AQI"]
existing_cols = [col for col in cols if col in df.columns]

for col in existing_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(y=df[col])
    plt.title(f"Box Plot of {col}")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()