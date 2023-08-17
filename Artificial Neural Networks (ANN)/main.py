import numpy as np
import pandas as pd
import tensorflow as tf

# Reading data from csv file
dataset = pd.read_csv('Churn_Modelling.csv')

# Creating matrix of features and dependent variable vector
X = dataset.iloc[:, 3:-1].values
Y = dataset.iloc[:, -1].values

# Encode gender
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
X = np.delete(X, 0, 1) # Dummy variable (n - 1) formula 

# Splitting dataset into training set and test set
from sklearn.model_selection import train_test_split

X_train , X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# Future Scaling (to every column) (compulsory)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))

# Adding the output layer                         "softmax" for > 3d
ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Compiling the ann                "categorical_crossentropy" > 3d
ann.compile(optimizer="adam", loss="binary_crossentropy" , metrics=["accuracy"])

# Training the ann
ann.fit(X_train, Y_train, batch_size=32, epochs=100)

# Predicting the result of single observation (in video)
"""
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Credit card? True
Active Member: True
Estimated Salary: 50000
"""
data = [1, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]
data = sc.transform([data])

prediction = ann.predict(data)

print(prediction, prediction > 0.5)

# Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_test, y_pred)
print(cm)
acc_score = accuracy_score(Y_test, y_pred)
print(acc_score)

