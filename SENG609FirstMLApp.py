#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Author:       Ethan Pattison
# FSU Course:   SENG 609
# Professor:    Dr Abusharkh
# Assingment:   Assignment 5: Building your first ML application
# Date:         9/21/2022


# In[708]:

# import packages and functions
import pandas as pd
import numpy as np
from numpy import array
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.compose import make_column_transformer
import joblib

# Link the Database to the Python file so the program has the information needed
# Save the Data as a dataframe

df = pd.read_csv('heart_2020.csv', sep=',')

df


# In[709]:


# Create Function to Change the Columns to Numeric Values.

def tran_HeartDisease(a):
    if a == "No":
        return 0
    if a == "Yes":
        return 1

    # Use the function and add the columns to the dataframe


df['HeartDiseaseNum'] = df['HeartDisease'].apply(tran_HeartDisease)
df['SmokingNum'] = df['Smoking'].apply(tran_HeartDisease)
df['AlcoholDrinkingNum'] = df['AlcoholDrinking'].apply(tran_HeartDisease)
df['StrokeNum'] = df['Stroke'].apply(tran_HeartDisease)
df['DiffWalkingNum'] = df['DiffWalking'].apply(tran_HeartDisease)
df['PhysicalActivityNum'] = df['PhysicalActivity'].apply(tran_HeartDisease)
df['AsthmaNum'] = df['Asthma'].apply(tran_HeartDisease)
df['KidneyDiseaseNum'] = df['KidneyDisease'].apply(tran_HeartDisease)
df['SkinCancerNum'] = df['SkinCancer'].apply(tran_HeartDisease)

df


# In[710]:


# Define a function to normalize

def normalize(Var):
    x = np.array(df[Var]).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler.fit(x)
    X_scaled = scaler.transform(x)
    df[Var] = X_scaled.reshape(1, -1)[0]


# Use the Function to normalize Physical and Mental Heath Ratings

normalize("MentalHealth")
normalize("PhysicalHealth")

# Print to See if the the values are changed
df.head(9)

# In[711]:


# Get the Columns and out from the file and save as a dataframe

XY = df[['Sex', 'HeartDiseaseNum', 'SmokingNum', 'AlcoholDrinkingNum', 'StrokeNum', 'PhysicalHealth', 'MentalHealth',
         'DiffWalkingNum', 'PhysicalActivityNum', 'SleepTime', 'AsthmaNum', 'KidneyDiseaseNum', 'SkinCancerNum']]

# In[712]:


# One Hot-Encode for the column Sex

columns_trans = make_column_transformer(
    (OneHotEncoder(), ["Sex"]),
    remainder='passthrough'
)
X = columns_trans.fit_transform(XY)
print(X)

# In[713]:


# Convert the Data back to a dataframe

Data = pd.DataFrame(X,
                    columns=['Female', 'Male', 'HeartDiseaseNum', 'SmokingNum', 'AlcoholDrinkingNum', 'StrokeNum',
                             'PhysicalHealth', 'MentalHealth',
                             'DiffWalkingNum', 'PhysicalActivityNum', 'SleepTime', 'AsthmaNum', 'KidneyDiseaseNum',
                             'SkinCancerNum'])
Data

# In[714]:


# Set the columns for X and Y

x = Data[['Male', 'Female', 'SmokingNum', 'AlcoholDrinkingNum', 'StrokeNum', 'PhysicalHealth', 'MentalHealth',
          'DiffWalkingNum', 'PhysicalActivityNum',
          'SleepTime', 'AsthmaNum', 'KidneyDiseaseNum', 'SkinCancerNum']]

y = Data[['HeartDiseaseNum']]

# Set the Training and the Testing
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

# Create a Linear Regression Model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Save the trained model to a file so we can use it to make prediction later
joblib.dump(model, 'HeartDisease.pkl')

# Report how well the model is performing
print("Model training results: ")

mse_train = mean_absolute_error(y_train, model.predict(X_train))
print(f" - Training Set Error: {mse_train}")

mse_test = mean_absolute_error(y_test, model.predict(X_test))
print(f" - Testing Set Error: {mse_test}")

# In[715]:


# Test The model by making a prediction

# Load in the model
model = joblib.load('HeartDisease.pkl')

# Define the inputs for person 1
Person_1 = [[0, 1, 1, 0, 0, 0.1, 1, 0, 1, 5, 1, 0, 1]]

# Make a prediction for the person
Heart_Prediction = model.predict(Person_1)

# We are prediction the first row in the array (0)
predicted_value = Heart_Prediction[0]

# print the results
print(f"Estimation for heart disease: {predicted_value}")


# In[716]:


def main():
    # Application to repeat the model for people that come into the clinic

    # Introduction to the program
    print("\nHello and welcome to the Healthy Heart Predictor!")

    print("\nThis program will ask the user 12 simple qustions and predict if the user is at risk for Heart disease: ")

    # Add Spacing
    print("\n")

    # Get all the information from the 12 Questions
    Attribute1 = int(input("\nAre you a male: (1 for Yes or 0 for No) "))
    Attribute3 = int(input("\nHave you smoked 100 cigarettes or more throughout your life: (1 for Yes or 0 for No) "))
    Attribute4 = int(input(
        "\nDo you average 1 alcoholic drink per day if female or 2 alcoholic drinks per day if male: (1 for Yes or 0 for No) "))
    Attribute5 = int(input("\nHave you ever been told you had a stroke: (1 for Yes or 0 for No) "))
    Attribute6 = int(
        input("\nHow many physical illness's and injury's have occured in the last Month: (Range is 0-30) "))
    Attribute7 = int(
        input("\nHow many days during the past 30 days was your mental health not good?: (Range is 0-30) "))
    Attribute8 = int(input("\nDo you have serious difficulty walking or climbing stairs?: (1 for Yes or 0 for No) "))
    Attribute9 = int(
        input("\nPhysical activity or exercise during the past 30 days other than their regular job:(Range is 0-30) "))
    Attribute10 = int(input("\nOn average, how many hours of sleep do you get in a 24-hour period?: "))
    Attribute11 = int(input("\nHave you ever been told you has asthma?: (1 for Yes or 0 for No) "))
    Attribute12 = int(input(
        "\nNot including kidney stones or bladder infections, were you ever told you had kidney disease? : (1 for Yes or 0 for No) "))
    Attribute13 = int(input("\nHave you ever been told you had skin cancer?: (1 for Yes or 0 for No) "))

    # Create a if statement to generate if Attribute 2 is female or not
    if Attribute1 == 1:
        Attribute2 = 0
    else:
        Attribute2 = 1

    # Add Spacing
    print("\n")

    # Load in the model
    model = joblib.load('HeartDisease.pkl')

    # Save the information from the person in a 2d array
    Person_1 = [[Attribute1, Attribute2, Attribute3, Attribute4, Attribute5,
                 Attribute6, Attribute7, Attribute8, Attribute9, Attribute10,
                 Attribute11, Attribute12, Attribute13]]

    # Make a prediction for the person
    Heart_Prediction = model.predict(Person_1)

    # We are prediction the first row in the array (0)
    predicted_value = Heart_Prediction[0]

    print("\nEstimation for Heart disease ranges from 0 to 1.")
    print("The Lower the estimation, the lower the chance you have of having or obtaining a heart disease.")

    # print the results
    print(f"\nYour Estimation for heart disease: {predicted_value}")

    if predicted_value >= .7:
        print(
            "\nYou are at a very high risk for Heart Disease, adjust lifestyle and see a medical professional as soon as possible.")

    elif predicted_value >= .5 and predicted_value < .7:
        print("\nYou are at risk for Heart Disease, please consider adjusting lifestyle")

    else:
        print("\nContinue to live the way you are, no adjustment need to be made.")
        print("Your body is at a low risk for Heart Disease.")

    restart = input("\nWould you like to analyze another person?: ").lower()
    if restart == "yes":
        print("\n")
        main()
    else:
        print("\nHave a great day!")


# In[700]:


# Run the main function

main()

# In[717]:


X = Data[['Male', 'Female', 'SmokingNum', 'AlcoholDrinkingNum', 'StrokeNum', 'PhysicalHealth', 'MentalHealth',
          'DiffWalkingNum', 'PhysicalActivityNum',
          'SleepTime', 'AsthmaNum', 'KidneyDiseaseNum', 'SkinCancerNum']]

Y = Data[['HeartDiseaseNum']]

# In[718]:


# Create the and display the Desision tree

from sklearn import tree

# Set the Training and the Testing
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

joblib.dump(clf, 'HeartDisease1.pkl')

# Report how well the model is performing
print("Classifier results: ")

mse_train = mean_absolute_error(y_train, clf.predict(X_train))
print(f" - Training Set Error: {mse_train}")

mse_test = mean_absolute_error(y_test, clf.predict(X_test))
print(f" - Testing Set Error: {mse_test}")

# In[719]:


# Example of a prediction
# Test The model by making a prediction

# Load in the model
Classifier = joblib.load('HeartDisease1.pkl')

# Define the inputs for person 1
Person_1 = [[0, 1, 1, 0, 0, 0.1, 1, 0, 1, 5, 1, 0, 1]]

# Make a prediction for the person
Heart_Prediction = model.predict(Person_1)

# We are prediction the first row in the array (0)
predicted_value = Heart_Prediction[0]

# print the results
print(f"Estimation for heart disease: {predicted_value}")


# In[720]:


def main_Function():
    # Application to repeat the model for people that come into the clinic

    # Introduction to the program
    print("\nHello and welcome to the Healthy Heart Predictor!")

    print("\nThis program will ask the user 12 simple qustions and predict if the user is at risk for Heart disease: ")

    # Add Spacing
    print("\n")

    # Get all the information from the 12 Questions
    Attribute1 = int(input("\nAre you a male: (1 for Yes or 0 for No) "))
    Attribute3 = int(input("\nHave you smoked 100 cigarettes or more throughout your life: (1 for Yes or 0 for No) "))
    Attribute4 = int(input(
        "\nDo you average 1 alcoholic drink per day if female or 2 alcoholic drinks per day if male: (1 for Yes or 0 for No) "))
    Attribute5 = int(input("\nHave you ever been told you had a stroke: (1 for Yes or 0 for No) "))
    Attribute6 = int(
        input("\nHow many physical illness's and injury's have occured in the last Month: (Range is 0-30) "))
    Attribute7 = int(
        input("\nHow many days during the past 30 days was your mental health not good?: (Range is 0-30) "))
    Attribute8 = int(input("\nDo you have serious difficulty walking or climbing stairs?: (1 for Yes or 0 for No) "))
    Attribute9 = int(
        input("\nPhysical activity or exercise during the past 30 days other than their regular job:(Range is 0-30) "))
    Attribute10 = int(input("\nOn average, how many hours of sleep do you get in a 24-hour period?: "))
    Attribute11 = int(input("\nHave you ever been told you has asthma?: (1 for Yes or 0 for No) "))
    Attribute12 = int(input(
        "\nNot including kidney stones or bladder infections, were you ever told you had kidney disease? : (1 for Yes or 0 for No) "))
    Attribute13 = int(input("\nHave you ever been told you had skin cancer?: (1 for Yes or 0 for No) "))

    # Create a if statement to generate if Attribute 2 is female or not
    if Attribute1 == 1:
        Attribute2 = 0
    else:
        Attribute2 = 1

    # Add Spacing
    print("\n")

    # Load in the model
    model = joblib.load('HeartDisease1.pkl')

    # Save the information from the person in a 2d array
    Person_1 = [[Attribute1, Attribute2, Attribute3, Attribute4, Attribute5,
                 Attribute6, Attribute7, Attribute8, Attribute9, Attribute10,
                 Attribute11, Attribute12, Attribute13]]

    # Make a prediction for the person
    Heart_Prediction = model.predict(Person_1)

    # We are prediction the first row in the array (0)
    predicted_value = Heart_Prediction[0]

    print("\nEstimation for Heart disease ranges from 0 to 1.")
    print("The Lower the estimation, the lower the chance you have of having or obtaining a heart disease.")

    # print the results
    print(f"\nYour Estimation for heart disease: {predicted_value}")

    if predicted_value >= .7:
        print(
            "\nYou are at a very high risk for Heart Disease, adjust lifestyle and see a medical professional as soon as possible.")

    elif predicted_value >= .5 and predicted_value < .7:
        print("\nYou are at risk for Heart Disease, please consider adjusting lifestyle")

    else:
        print("\nContinue to live the way you are, no adjustment need to be made.")
        print("Your body is at a low risk for Heart Disease.")

    restart = input("\nWould you like to analyze another person?: ").lower()
    if restart == "yes":
        print("\n")
        main_Function()
    else:
        print("\nHave a great day!")


# In[721]:


# Run the main function

main_Function()

