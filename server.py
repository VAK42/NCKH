from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
app = FastAPI(title="Stress Inference Service")
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
negWords = ['căng thẳng', 'mệt', 'áp lực', 'lo lắng', 'sợ', 'chán', 'khó', 'stress', 'buồn', 'tệ', 'kém', 'thất bại', 'không thể', 'quá tải', 'kiệt sức', 'không hiểu', 'khó khăn', 'nặng nề', 'mất ngủ', 'lo âu']
posWords = ['vui', 'tốt', 'ổn', 'thoải mái', 'tự tin', 'thích', 'hứng thú', 'hiểu', 'dễ', 'hài lòng', 'tích cực', 'yêu thích', 'hứng khởi']
def getSentimentScore(textValue):
  if not textValue: return 0
  textValue = str(textValue).lower()
  negCount = sum(1 for word in negWords if word in textValue)
  posCount = sum(1 for word in posWords if word in textValue)
  return negCount - posCount
class InferenceRequest(BaseModel):
  responseTime: float
  lateCount: int
  lmsAccess: float
  accessTiming: int
  emotionalText: str
  sleepHours: float
  sleepQuality: int
  procrastinationLevel: int
  interactionScore: int
@app.post("/predict")
async def predictStress(requestInfo: InferenceRequest):
  try:
    sentimentValue = getSentimentScore(requestInfo.emotionalText)
    healthIndex = requestInfo.sleepHours * requestInfo.sleepQuality
    digitalInteraction = requestInfo.lmsAccess * requestInfo.interactionScore
    featureArray = [
      requestInfo.responseTime, requestInfo.lateCount, requestInfo.lmsAccess,
      requestInfo.accessTiming, sentimentValue, requestInfo.sleepHours,
      requestInfo.procrastinationLevel, healthIndex, digitalInteraction
    ]
    inputData = np.array(featureArray).reshape(1, -1)
    scaledData = scaler.transform(inputData)
    prediction = model.predict(scaledData)[0]
    prob = model.predict_proba(scaledData)[0] if hasattr(model, "predict_proba") else None
    return {
      "Stress Level": "High Stress" if prediction == 1 else "Normal",
      "Prediction Code": int(prediction),
      "Confidence Score": float(max(prob)) if prob is not None else 1.0
    }
  except Exception as error:
    raise HTTPException(status_code=500, detail=str(error))
if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000)