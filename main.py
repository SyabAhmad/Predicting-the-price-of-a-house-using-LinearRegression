print("Linear Regression")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

data = pd.read_csv("Data/HousePriceIndia.csv")
# # print(data.head(10))
# print(data.columns)
#
# # to check if the data contains Null values
# print(data.isnull().sum())

xData = data[['number of bedrooms',
             'number of bathrooms',
             'living area',
             'lot area',
             'number of floors',
             'waterfront present',
             'number of views',
             'condition of the house',
             'Area of the house(excluding basement)',
             'Area of the basement',
             'Built Year',
             'Renovation Year',
             'Postal Code',
             'Lattitude',
             'Longitude',
             'living_area_renov',
             'lot_area_renov',
             'Number of schools nearby',
             'Distance from the airport']]

yData = data['Price']

xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=20, random_state=32)

toFit = LinearRegression()

toFit.fit(xTrain, yTrain)

yPrediction = toFit.predict(xTest)

fitingModel = pd.DataFrame({'Actual Price': yTest, 'Prediction Price': yPrediction})

print(fitingModel.head(5))


accuracy = toFit.score(xTest,yTest)
print("Accuracy 1",accuracy)

mse = np.mean((yPrediction - yTest)**2)
print("Mean Squared Error:", mse)