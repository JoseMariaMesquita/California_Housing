"""Plantilla: predicción del precio de casas en California.

Completa las partes marcadas con TODO. No cambies la estructura general.
Autores: SanchezL, MesquitaJ
"""
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree

TARGET = "MedHouseVal"
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_DEPTH = 3
MODEL_PATH = "modelo_california.pkl"


def check_nulls(df: pd.DataFrame):
    print("Valores nulos por columna:")
    print(df.isnull().sum())


def handle_nulls(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()

def plot_decision_tree(model, max_depth=None):
    DEFAULT_MAX_DEPTH = 5
    MAX_REASONABLE_DEPTH = 20

    if max_depth is None or max_depth > MAX_REASONABLE_DEPTH:
        max_depth = DEFAULT_MAX_DEPTH

    plt.figure(figsize=(20, 10))

    plot_tree(
        model,
        max_depth=max_depth,
        filled=True,
        feature_names=model.feature_names_in_
    )

    plt.savefig("arbol_decision.png", bbox_inches="tight")
    plt.close()



def compute_errors(y_true, y_pred, print_errors: bool = True) -> tuple[float, float]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    if print_errors:
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
    return mae, mse


def save_model(model, path: str = MODEL_PATH):
    joblib.dump(model, path)
    print(f"Modelo guardado en: {path}")


def validate_data():
    df = fetch_california_housing(as_frame=True).frame
    check_nulls(df)
    df = handle_nulls(df)


def build_model():
    df = fetch_california_housing(as_frame=True).frame

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = DecisionTreeRegressor(max_depth=MAX_DEPTH, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    compute_errors(y_test, y_pred)

    return model


def plot_data():
    model = build_model()
    plot_decision_tree(model, max_depth=MAX_DEPTH)


def main():
    model = build_model()
    save_model(model)


if __name__ == "__main__":
    validate_data()
    plot_data()
    main()

#Jose Maria Mesquita