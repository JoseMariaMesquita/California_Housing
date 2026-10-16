"""Plantilla: servicio web para el modelo de precios de casas en California.

Solo tienes que completar las partes marcadas con TODO.
El resto del código ya está hecho.
Autores: <nombre1>, <nombre2>

Ejecución en local (con modelo_california.pkl generado en el ejercicio 1):
    uvicorn app:app --reload
Comando de arranque en Render (Start Command):
    uvicorn app:app --host 0.0.0.0 --port $PORT
"""
import os
from contextlib import asynccontextmanager

import joblib
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# Ruta local del modelo.
MODEL_PATH = "modelo_california.pkl"

# URL directa del modelo almacenado en GitHub.
MODEL_URL = (
    "https://raw.githubusercontent.com/"
    "JoseMariaMesquita/California_Housing/main/modelo_california.pkl"
)

DOWNLOAD_TIMEOUT = 30


# Estado de la aplicación.
state = {"model": None}


class HouseFeatures(BaseModel):
    """Variables de entrada del modelo."""

    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


def download_model(url: str = MODEL_URL, path: str = MODEL_PATH):
    """
    Descarga el modelo desde GitHub y lo guarda localmente.
    """

    print(f"Descargando el modelo desde GitHub: {url}")

    response = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
    response.raise_for_status()

    with open(path, "wb") as file:
        file.write(response.content)

    print(f"Modelo descargado correctamente en {path}")


def load_model(path: str = MODEL_PATH):
    """
    Carga el modelo local.
    Si no existe, lo descarga desde GitHub antes de cargarlo.
    """

    if not os.path.exists(path):
        print(f"El archivo {path} no existe localmente.")
        print("Iniciando descarga desde GitHub...")

        download_model(path=path)

    print("Cargando el modelo...")
    model = joblib.load(path)

    print("Modelo cargado correctamente.")

    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Carga el modelo al iniciar el servicio.
    """

    state["model"] = load_model()
    yield


app = FastAPI(
    title="Precio de casas en California",
    lifespan=lifespan,
)


@app.get("/")
def health():
    """Comprueba que el servicio está funcionando."""

    return {"status": "ok"}


@app.post("/predict")
def predict(features: HouseFeatures):
    """Predice el precio medio de una vivienda."""

    model = state["model"]

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no cargado",
        )

    X = pd.DataFrame([features.model_dump()])
    prediction = model.predict(X)[0]

    return {"MedHouseVal": float(prediction)}




