# Prediction of price of the houses

### First import all required imports
```python
print("Linear Regression")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
```
### Read and display Data
```python
data = pd.read_csv("Data/HousePriceIndia.csv")
# print(data.head(10))
print(data.columns)
```


### to check if the data contains Null values
```python
print(data.isnull().sum())
```
### spliting the data to xData and yData
```python
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
```
### To Train and Test the model

```python
xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=20, random_state=32)

toFit = LinearRegression()

toFit.fit(xTrain, yTrain)

yPrediction = toFit.predict(xTest)

fitingModel = pd.DataFrame({'Actual Price': yTest, 'Prediction Price': yPrediction})

```
### To display the Prediction data
```python
print(fitingModel.head(5))
```


### To check the accuracy and remove the mean square error
```python
accuracy = toFit.score(xTest,yTest)
print("Accuracy 1",accuracy)

mse = np.mean((yPrediction - yTest)**2)
print("Mean Squared Error:", mse)
```

```output
       Actual Price  Prediction Price
4054         219950     253924.752220
11653        513000     548970.264743
2200         405000     342701.126597
9134         425000     301027.971400
10540        411000     723325.198149
Accuracy 1 0.8037809187152187
Mean Squared Error: 16560868221.700201

```
The accuracy is 80% and not good enough we will try another algorithm which is called Decision Tree Algorithm.
it is simple first we will import the tree from sklearn then we will fit it and predict.

So after checking all the 4 Algorithms (as below) the output of all 4 are below
```
 ✔ Linear Regression
 ✔ Decision Tree Algorithm
 ✔ Random Forests Algorithm
 ✔ Gradient Boosting Algorithm
```

### ✔ Linear Regression
```output
       Actual Price  Prediction Price
4054         219950     253924.752220
11653        513000     548970.264743
2200         405000     342701.126597
9134         425000     301027.971400
10540        411000     723325.198149
Accuracy 1 0.8037809187152187
Mean Squared Error: 16560868221.700201
```

### ✔ Decision Tree Algorithm
```output
       Actual Price  Prediction Price
4054         219950          216000.0
11653        513000          540000.0
2200         405000          570000.0
9134         425000          415000.0
10540        411000          450000.0
Accuracy 1 0.6789019212161975
Mean Squared Error: 27100641457.3
```

### ✔ Random Forests Algorithm
```output
       Actual Price  Prediction Price
4054         219950         245223.03
11653        513000         541943.37
2200         405000         479770.90
9134         425000         443957.88
10540        411000         474652.37
Accuracy :  0.8702546741018636
Mean Squared Error: 10950490800.953691
```

### ✔ Gradient Boosting Algorithm
```output
       Actual Price  Prediction Price
4054         219950     253218.818394
11653        513000     525831.272862
2200         405000     431644.948290
9134         425000     428246.838995
10540        411000     488766.121402
Accuracy :  0.7306688761169409
Mean Squared Error: 22731516330.749928
```
All 4 regression algorithm completed and the best one to fit this datasets is "Random Forest Algorithm"

if this repository helps you then follow me for more learning materials like this,

```"""about developer"""```

```Name: Syab Ahmad```

```Tools: Pycharm```

```Language: Python```
