import pandas as pd
import numpy as np

df = pd.read_csv("titanic.csv")

print("Original Shape")
print(df.shape)

df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df['Fare'] >= lower) & (df['Fare'] <= upper)]

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = np.where(df['FamilySize'] == 1, 1, 0)
df['AgeFareRatio'] = df['Age'] / (df['Fare'] + 1)

print("\nCleaned Shape")
print(df.shape)

df.to_csv("cleaned_titanic.csv", index=False)

print("\nProject Completed Successfully")