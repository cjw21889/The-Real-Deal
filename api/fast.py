from datetime import datetime

import pandas as pd
import joblib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?


@app.get("/")
def index():
    return dict(greeting="hello")


@app.get("/predict")
def predict(year, rated, runtime, genre, director, writer, actors, plot, language, production):

    # build X ⚠️ beware to the order of the parameters ⚠️
    X = pd.DataFrame(dict(
        Year=[year],
        Rated=[rated],
        Runtime=[runtime],
        Genre=[genre],
        Director=[director],
        Writer=[writer],
        Actors=[actors],
        Plot=[plot],
        Language=[language],
        Production=[production]
        ))

    # pipeline = get_model_from_gcp()
    pipeline = joblib.load('model.joblib')

    # make prediction
    results = pipeline.predict(X)

    # convert response from numpy to python type
    pred = float(results[0])

    return dict(prediction=pred)

