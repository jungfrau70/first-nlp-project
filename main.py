import uvicorn
from fastapi import FastAPI

import joblib

gender_vectorizer = open("models/gender_vectorizer.pkl", 'rb')
gender_cv = joblib.load(gender_vectorizer)

model_nv = open("models/gender_nv_model.pkl", 'rb')
gender_clf = joblib.load(model_nv)

app = FastAPI()

@app.get('/')
async def index():
    return { 'message': "안녕하세요" }

@app.get('/items/{name}')
async def get_item(name:str):
    return { 'name': name }

@app.get('/predict/{name}')
async def predict(name):
    vectorized_name = gender_cv.transform([name]).toarray()
    prediction = gender_clf.predict(vectorized_name)

    if prediction[0] == 0:
        result = "여성"
    else:
        result = '남성'
    
    return { 'origin name': name, '예측': result }

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)