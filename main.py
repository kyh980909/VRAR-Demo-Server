import tensorflow as tf
from tensorflow.keras.models import load_model

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

from sklearn.preprocessing import scale
from PIL import Image

import uvicorn
from feature_extractor import FeatureExtractor
import numpy as np
import pandas as pd
from pathlib import Path

from fastapi import FastAPI, UploadFile, Request

from pydantic import BaseModel

rssi_col = pd.read_csv('./static/rssi_col.csv')
rssi_model = load_model('./static/model/radio_map_model')

# Read image features
fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in Path("./static/feature").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
features = np.array(features)

app = FastAPI()

@app.get("/")
async def Hello():
    return {"Hello": "Demo Simulator"}

@app.post("/visual_map_predict/feature_extract")
async def visual_map_predict(image: UploadFile):
    # uploaded_img_path = f"static/uploaded/{image.filename}"
    img = Image.open(image.file)
    # Run search
    query = fe.extract(img)
    dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
    dists = dists.tolist()
    ids = np.argsort(dists)[:5]  # Top 5 results
    ids = ids.tolist()
    scores = [(dists[id], img_paths[id]) for id in ids]
    return {"result": scores}

@app.post("/visual_map_predict/feature_match")
async def visual_map_predict(image: UploadFile):
    # uploaded_img_path = f"static/uploaded/{image.filename}"
    img = Image.open(image.file)
    # Run search
    
    return {"result": "feature match"}

@app.post("/radio_map_predict")
async def radio_map_predict(rssi: Request):
    rssi_dummy = pd.DataFrame(index=range(0,0), columns=list(rssi_col.columns))
    rssi = await rssi.json()
    rssi_input = pd.concat([rssi_dummy, pd.DataFrame([rssi['rssi']])])

    drop_column_list = list(set(list(rssi_input.columns)) - set(list(rssi_col.columns)))
    if len(drop_column_list) != 0: 
        rssi_input = rssi_input.drop(drop_column_list, axis=1) 

    rssi_input = rssi_input.replace(np.nan, -110)
    rssi_input = scale(np.asarray(rssi_input).astype(float), axis=1)

    result = rssi_model.predict(np.expand_dims(rssi_input, axis=0)) ## predict에 들어갈 값

    print(np.argmax(result))
    return json.dumps({"result": int(np.argmax(result)) })