from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from fuzzywuzzy import fuzz

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Load the model
model = load_model('C:/Users/delic/Desktop/taslEvMl/taskEvolt/application_back/model.h5')
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#poziva ann trenirani model i na osnovu njega donosi odluku
@app.post('/predict/ann')
async def predict(productOne: str, productTwo: str):
    try:
        #print(productOne, productTwo)
        productOneEmbeddings = embedding_model.encode(productOne)
        productTwoEmbeddings = embedding_model.encode(productTwo)
        
        # Najproblematičnija linija tokom pisanja koda, obzirom da rad sa spajanjem embedinga je očekivao liste a obični stringovi su u početku proslijeđeni       
        productPairEmbeddings = np.concatenate(([productOneEmbeddings], [productTwoEmbeddings]), axis=1)
        predictions = model.predict(productPairEmbeddings)
        #predikcije su vraćale izvorno matricu. ista se nije mogla direktno vratiti. potrebno je izvući tačnu vrijednost i potom je striktno u float pretvorit. 
        return {'prediction': float(predictions[0][0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# zanimljiv poziv radi komparacije, koristenje samo uvezenih embeddinga koji potom nisu trenirani nad datasetom. vraća sličnost na osnovu uvezenih emdeddinga između riječi
@app.post('/predict/just/embedding')
async def predict(productOne: str, productTwo: str):
    try:
        productOneEmbeddings = embedding_model.encode(productOne)
        productTwoEmbeddings = embedding_model.encode(productTwo)
        
        similarities = embedding_model.similarity(productOneEmbeddings, productTwoEmbeddings)
        # povratna vrijednsot sličnosti je bila problematićna, obzirom da je vraćala tenzor. Stoga najviše vremena u ovoj funkciji je utrošeno na način pretvaranja tenzora. 
        logging.info("Similarity:", similarities)
        
        return {'Similarity': similarities.item()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# biran je token set ratio na osnovu njegove definicije iz priloga koja je preuzeta sa: https://www.analyticsvidhya.com/blog/2021/07/fuzzy-string-matching-a-hands-on-guide/
# Token set ratio performs a set operation that takes out the common tokens instead of just tokenizing the strings, 
# sorting, and then pasting the tokens back together. Extra or same repeated words do not matter.
# Dakle bolja poklapanja za naš slučaj se mogu izabrati ovom funkcijom u odnosu na ostale (primjer kad se gledaju sličnosti riječi jedna za drugom); 

@app.post('/predict/fuzzywuzzy')
async def predict(productOne: str, productTwo: str):
    try:
        result = fuzz.token_set_ratio(productOne, productTwo)
        
        return {'similarity': result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

