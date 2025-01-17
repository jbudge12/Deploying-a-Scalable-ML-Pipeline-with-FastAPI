import pytest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dummy data defined in the DF below for test purposes. 
def load_data():
    """Loads the data from a CSV file or database."""
    data = pd.DataFrame({
        "age": [30, 18, 22, 27],
        "workclass": ["Private", "Self-emp", "Private", "Gov"],
        "education-num": [10, 11, 13, 14],
        "hours-per-week": [40, 50, 60, 80],
        "salary": [0, 1, 0, 1]
    })
    return data

def train_model(X_train, y_train):
    #Train the logistic model similarly to before
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model

def compute_metrics(model, X_test, y_test):
    #Get model accuracy
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)

# Unit tests
#Test 1
def test_data_load():
    #Test if the data load returns the expected columns
    data = load_data()
    expected_columns = ["age", "workclass", "education-num", "hours-per-week", "salary"]
    assert isinstance(data, pd.DataFrame), "Data is not a DataFrame."
    assert list(data.columns) == expected_columns, "Data columns do not match expected columns."
    assert not data.isnull().values.any(), "Data contains null values."

#Test 2
def test_model_training():
    #Test to ensure the model is the correct type
    data = load_data()
    X = data[["age", "education-num", "hours-per-week"]]
    y = data["salary"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = train_model(X_train, y_train)
    assert isinstance(model, LogisticRegression), "Trained model isn't a LogisticRegression model."


#Test3
def test_metrics_computation():
    #Test to ensure accuracy is returned appropriately
    data = load_data()
    X = data[["age", "education-num", "hours-per-week"]]
    y = data["salary"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = train_model(X_train, y_train)
    accuracy = compute_metrics(model, X_test, y_test)
    assert 0 <= accuracy <= 1, "Accuracy is not within the valid range of 0 to 1."
