
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def create_logistic_model():
    """
    Create a Logistic Regression pipeline.

    StandardScaler scales the input features.
    LogisticRegression performs binary classification.
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=42
        ))
    ])

    return model
