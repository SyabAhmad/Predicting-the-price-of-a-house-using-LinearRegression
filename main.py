print("Linear Regression")
import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv("Data/HousePriceIndia.csv")
print(data.head(10))

print(data.columns)