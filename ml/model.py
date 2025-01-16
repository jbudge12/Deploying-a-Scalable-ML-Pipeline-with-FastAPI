import pickle
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from ml.data import process_data


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model : LogisticRegression
        Trained machine learning model.
    """
    model = LogisticRegression()
    model.fit(X_train, y_train)  # Train the model
    return model  # Return the trained model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """
    Runs model inferences and returns the predictions.

    Inputs
    ------
    model : LogisticRegression
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def save_model(model, path):
    """
    Serializes model to a file.

    Inputs
    ------
    model : object
        Trained machine learning model or encoder.
    path : str
        Path to save the pickle file.
    """
    with open(path, 'wb') as file:
        pickle.dump(model, file)


def load_model(path):
    """
    Loads pickle file from `path` and returns it.

    Inputs
    ------
    path : str
        Path to load the pickle file from.
    Returns
    -------
    model : object
        The deserialized model or encoder.
    """
    with open(path, 'rb') as file:
        return pickle.load(file)


def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features, label, encoder, lb, model
):
    """
    Computes the model metrics on a slice of the data specified by a column name.

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing the features and label.
    column_name : str
        Column containing the sliced feature.
    slice_value : str, int, or float
        Value of the slice feature.
    categorical_features: list
        List containing the names of the categorical features.
    label : str
        Name of the label column in the data.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer.
    model : LogisticRegression
        Trained machine learning model.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    # Filter data for the slice
    data_slice = data[data[column_name] == slice_value]

    # Process the data slice for inference
    X_slice, y_slice, _, _ = process_data(
        data_slice,
        categorical_features=categorical_features,
        label=label,
        encoder=encoder,
        lb=lb,
        training=False
    )

    # Get predictions for the slice
    preds = inference(model, X_slice)

    # Compute metrics
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta
