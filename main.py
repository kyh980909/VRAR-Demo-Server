from datetime import datetime
import json
from PIL import Image

import uvicorn
from feature_extractor import FeatureExtractor
import numpy as np
from pathlib import Path

from fastapi import FastAPI, UploadFile

from pydantic import BaseModel

# Read image features
fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in Path("./static/feature").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
features = np.array(features)

class RadioMap(BaseModel):
    rp : int

app = FastAPI()

@app.get("/")
async def Hello():
    return {"Hello": "1"}
    
@app.get("/items/{item_id}")
async def Hello(item_id: int):
    return {"item_id": item_id}

@app.post("/visual_map_predict")
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

# if __name__ == "__main__":
#     uvicorn.run(app, port=15261, host='0.0.0.0', reload=True)