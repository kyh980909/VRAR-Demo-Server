from typing import List, Optional
import tensorflow as tf
from tensorflow.keras.models import load_model

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

from sklearn.preprocessing import scale
from PIL import Image

import base64
import io
import json

from feature_extractor import FeatureExtractor
import numpy as np
import pandas as pd
from pathlib import Path
from pydantic import BaseModel

from fastapi import FastAPI, UploadFile, Request

rssi_col = pd.read_csv('./static/rssi_col.csv')
rssi_model = load_model('./static/model/radio_map_model')

# Read image features
fe = FeatureExtractor()

def from_image_to_bytes(img):
    """
    PIL image 객체를 bytes로 변환
    """
    imageByteArr = io.BytesIO()
    img.save(imageByteArr, format=img.format)

    encoded = base64.b64encode(imageByteArr.getvalue())
    decoded = encoded.decode('ascii')

    return decoded

app = FastAPI()

# class VisualSearch(BaseModel):
#     image: UploadFile
#     rssi: str = None

class RSSI(BaseModel):
    rp: List[int]
    prob : List[float]


@app.get("/")
async def Hello():
    return {"Hello": "Demo Simulator"}

@app.post("/visual_map_predict/feature_extract")
async def visual_map_predict(image: UploadFile, rp:str):
    # uploaded_img_path = f"static/uploaded/{image.filename}"
    print(rp)
    features = np.load(f'./static/feature/{rp}.npy', allow_pickle=True)
    img_paths = np.load(f'./static/feature/{rp}_path.npy', allow_pickle=True)

    img = Image.open(image.file)
    # Run search
    query = fe.extract(img)

    dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
    dists = dists.tolist()
    ids = np.argsort(dists)[0]  # Top 5 results
    ids = ids.tolist()
    dist, img_path = (dists[ids], img_paths[ids])

    search_img = Image.open(img_path)
    search_img = from_image_to_bytes(search_img)
    return {"img":search_img, "dist":str(round(dist, 4)), "rp":str(img_path).split("\\")[2].split("_")[0]}

@app.post("/visual_map_predict/feature_match")
async def visual_map_predict(image: UploadFile):
    # uploaded_img_path = f"static/uploaded/{image.filename}"
    img = Image.open(image.file)
    # Run search
    
    return {"result": "feature match"}

@app.post("/radio_map_predict", response_model=RSSI)
async def radio_map_predict(rssi: Request, max_rp: int):
    rssi_dummy = pd.DataFrame(index=range(0,0), columns=list(rssi_col.columns))
    rssi = await rssi.json()
    rssi_input = pd.concat([rssi_dummy, pd.DataFrame([rssi['rssi']])])

    drop_column_list = list(set(list(rssi_input.columns)) - set(list(rssi_col.columns)))
    if len(drop_column_list) != 0: 
        rssi_input = rssi_input.drop(drop_column_list, axis=1) 

    rssi_input = rssi_input.replace(np.nan, -110)
    rssi_input = scale(np.asarray(rssi_input).astype(float), axis=1)

    result = rssi_model.predict(np.expand_dims(rssi_input, axis=0)) ## predict에 들어갈 값
    result = np.squeeze(result[0])
    ids = np.argsort(result)[::-1]

    prob = list(result[ids[:max_rp]])
    return {"rp": list(ids[:max_rp]), "prob":prob}