import requests
import json
def sendPredictionRequest():
  targetUrl = "http://127.0.0.1:8000/predict"
  studentInfo = {
    "responseTime": 5.2,
    "lateCount": 2,
    "lmsAccess": 15.0,
    "accessTiming": 2,
    "emotionalText": "Tôi đang cảm thấy áp lực vì khối lượng bài tập quá lớn.",
    "sleepHours": 5.0,
    "sleepQuality": 2,
    "procrastinationLevel": 4,
    "interactionScore": 3
  }
  try:
    response = requests.post(targetUrl, json=studentInfo)
    if response.status_code == 200:
      predictionResult = response.json()
      print("\n--- Prediction Result Dashboard ---")
      for key, value in predictionResult.items():
        print(f"{key}: {value}")
      print("----------------------------------\n")
    else:
      print(f"Error: {response.text}")
  except Exception as error:
    print(f"Connection Error: {error}")
if __name__ == "__main__":
  sendPredictionRequest()