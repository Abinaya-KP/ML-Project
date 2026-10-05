import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("train.csv")

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

le_sex = LabelEncoder()
le_embarked = LabelEncoder()

df["Sex"] = le_sex.fit_transform(df["Sex"])
df["Embarked"] = le_embarked.fit_transform(df["Embarked"])

# Features and target
X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
y = df["Survived"]

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

print("\n======================================")
print("     TITANIC SURVIVAL PREDICTION")
print("======================================")

print("\nEnter the Passenger Details")

pclass = int(input("Passenger Class (1/2/3): "))
sex = input("Sex (Male/Female): ").strip().lower()
age = float(input("Age: "))
sibsp = int(input("Number of Siblings/Spouses: "))
parch = int(input("Number of Parents/Children: "))
fare = float(input("Fare: "))
embarked = input("Embarked (C/Q/S): ").strip().upper()

if sex == "male":
    sex = le_sex.transform(["male"])[0]
else:
    sex = le_sex.transform(["female"])[0]

embarked = le_embarked.transform([embarked])[0]

passenger = pd.DataFrame([[
    pclass,
    sex,
    age,
    sibsp,
    parch,
    fare,
    embarked
]], columns=[
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
])

prediction = model.predict(passenger)[0]

# Display result
print("\n======================================")

if prediction == 1:
    print("Prediction: SURVIVED")
else:
    print("Prediction: NOT SURVIVED")

print("======================================")