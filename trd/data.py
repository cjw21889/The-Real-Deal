import os
import pandas as pd
import numpy as np
import re


def load_data(n, movies_only=True):
    df = pd.read_csv("raw_data/merged_movies_by_index.csv", nrows=n)
    if movies_only:
        df = df[df['Type'] != 'series']

    return df
