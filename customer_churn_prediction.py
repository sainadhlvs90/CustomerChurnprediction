import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
data = pd.read_csv(
    "customer_churn.csv"
)

print(data.head())
print("\nDataset Info:\n")

print(data.info())
print("\nMissing Values:\n")

print(data.isnull().sum())
encoder = LabelEncoder()

data["ContractType"] = encoder.fit_transform(
    data["ContractType"]
)

data["InternetService"] = encoder.fit_transform(
    data["InternetService"]
)

data["Churn"] = encoder.fit_transform(
    data["Churn"]
)

print("\nEncoded Data:\n")

print(data.head())
sns.countplot(
    x="Churn",
    data=data
)

plt.title(
    "Customer Churn Count"
)

plt.show()
X = data.drop(
    "Churn",
    axis=1
)

y = data["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
model = LogisticRegression()

model.fit(
    X_train,
    y_train
)
predictions = model.predict(
    X_test
)
accuracy = accuracy_score(
    y_test,
    predictions
)

print(
    "\nModel Accuracy:",
    accuracy * 100
)
new_customer = np.array([
    [30,600,0,15,1]
])

prediction = model.predict(
    new_customer
)

if prediction[0] == 1:
    print(
        "\nCustomer May Leave"
    )
else:
    print(
        "\nCustomer Will Stay"
    )