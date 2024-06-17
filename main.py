from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Load the model
model = load_model('C:/Users/delic/Desktop/taslEvMl/taskEvolt/application_back/model.h5')
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


@app.post('/predict')
async def predict(productOne: str, productTwo: str):
    try:
        #print(productOne, productTwo)
        productOneEmbeddings = embedding_model.encode(productOne)
        productTwoEmbeddings = embedding_model.encode(productTwo)
        
        #bez ovog javlja error dimenzija Exception encountered when calling Sequential.call().\n\n\u001b[1mInvalid input shape for
        #input Tensor("data:0", shape=(32,), dtype=float32). Expected shape (None, 768), but input has incompatible shape (32,)

        productOneEmbeddings = np.expand_dims(productOneEmbeddings, axis=0)
        productTwoEmbeddings = np.expand_dims(productTwoEmbeddings, axis=0)
        
        
        productPairEmbeddings = np.concatenate((productOneEmbeddings, productTwoEmbeddings), axis=1)
        predictions = model.predict(productPairEmbeddings)
        return {'prediction': predictions}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

