import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-whitegrid')
filePath = "NCKH.csv"
df = pd.read_csv(filePath)
df.columns = df.columns.str.strip()
colMap = {
  'Trung bình bạn mất bao lâu để bắt đầu làm bài tập sau khi được giao (tính bằng giờ)?': 'responseTime',
  'Trong 4 tuần gần nhất, bạn nộp muộn bao nhiêu lần?': 'lateCount',
  'Bạn truy cập hệ thống học tập (LMS, portal lớp) trung bình bao nhiêu lần mỗi ngày?': 'lmsAccess',
  'Hãy mô tả cảm xúc hiện tại của bạn về việc học tập (2–5 câu, 50–200 từ):': 'emotionalText',
  'Mức độ căng thẳng do học tập trong 2 tuần gần nhất': 'stressScore',
  'Bạn ngủ trung bình bao nhiêu giờ mỗi đêm?': 'sleepHours',
  'Bạn có thường trì hoãn khi làm bài hoặc học không?': 'procrastinationLevel',
  'Chất lượng giấc ngủ trong 2 tuần gần đây': 'sleepQuality',
  'Mức độ tương tác của bạn trong lớp học (phát biểu, hỏi giảng viên, thảo luận nhóm)?': 'interactionScore'
}
dfClean = df.rename(columns=colMap)[list(colMap.values())]
def cleanHours(text):
  if pd.isna(text): return np.nan
  text = str(text).lower()
  if any(w in text for w in ['ngay', 'luôn', 'lập tức']): return 0.1
  nums = re.findall(r'\d+(?:[.,]\d+)?', text)
  if not nums: return np.nan
  val = float(nums[0].replace(',', '.'))
  if 'ngày' in text or 'day' in text: return val * 24
  if 'phút' in text or 'p' in text or "'" in text: return val / 60
  return val
def cleanRange(text):
  if pd.isna(text): return 0
  text = str(text)
  nums = re.findall(r'\d+', text)
  if not nums: return 0
  nums = [float(n) for n in nums]
  return sum(nums) / len(nums)
def cleanLateCount(text):
  if pd.isna(text): return 0
  nums = re.findall(r'\d+', str(text))
  return int(nums[0]) if nums else 0
dfClean['responseTime'] = dfClean['responseTime'].apply(cleanHours)
dfClean['responseTime'] = pd.to_numeric(dfClean['responseTime'], errors='coerce')
dfClean['responseTime'] = dfClean['responseTime'].fillna(dfClean['responseTime'].median())
dfClean['sleepHours'] = dfClean['sleepHours'].apply(cleanHours)
dfClean['sleepHours'] = pd.to_numeric(dfClean['sleepHours'], errors='coerce')
dfClean['sleepHours'] = dfClean['sleepHours'].fillna(dfClean['sleepHours'].median())
dfClean['lmsAccess'] = dfClean['lmsAccess'].apply(cleanRange)
dfClean['lateCount'] = dfClean['lateCount'].apply(cleanLateCount)
dfClean['stressScore'] = pd.to_numeric(dfClean['stressScore'], errors='coerce')
dfClean['stressScore'] = dfClean['stressScore'].fillna(dfClean['stressScore'].median())
vectorizer = TfidfVectorizer(max_features=50, stop_words=['là', 'và', 'của', 'thì'])
tfidfMatrix = vectorizer.fit_transform(dfClean['emotionalText'].fillna("").astype(str))
dfClean['sentimentIntensity'] = tfidfMatrix.sum(axis=1)
dfClean['healthIndex'] = dfClean['sleepHours'] * dfClean['sleepQuality']
dfClean['digitalInteraction'] = dfClean['lmsAccess'] * dfClean['interactionScore']
features = ['responseTime', 'lateCount', 'lmsAccess', 'sentimentIntensity', 'sleepHours', 'procrastinationLevel', 'healthIndex', 'digitalInteraction']
target = 'stressScore'
dfFinal = dfClean.dropna(subset=features + [target])
X = dfFinal[features]
y = dfFinal[target]
scaler = StandardScaler()
xScaled = scaler.fit_transform(X)
xTrain, xTest, yTrain, yTest = train_test_split(xScaled, y, test_size=0.2, random_state=42)
print("=== Kết Quả Phân Tích Dữ Liệu NCKH ===\n")
linReg = LinearRegression()
linReg.fit(xTrain, yTrain)
yPredLin = linReg.predict(xTest)
print("1. Mô Hình Tuyến Tính (Linear Regression)")
print(f"   - MSE: {mean_squared_error(yTest, yPredLin):.4f}")
print(f"   - R2 Score: {r2_score(yTest, yPredLin):.4f}")
yClass = (y >= 4).astype(int)
xTrainC, xTestC, yTrainC, yTestC = train_test_split(xScaled, yClass, test_size=0.2, random_state=42)
dtModel = DecisionTreeClassifier(max_depth=4, random_state=42)
dtModel.fit(xTrainC, yTrainC)
yPredDt = dtModel.predict(xTestC)
print("\n2. Mô Hình Cây Quyết Định (Classification)")
print(f"   - Độ Chính Xác: {accuracy_score(yTestC, yPredDt):.4f}")
print(classification_report(yTestC, yPredDt, target_names=['Bình Thường', 'Stress Cao']))
plt.figure(figsize=(10, 8))
sns.heatmap(dfFinal[features + [target]].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Ma Trận Tương Quan (Correlation Matrix)")
plt.tight_layout()
plt.show()
plt.figure(figsize=(15, 10))
plot_tree(dtModel, feature_names=features, class_names=['Normal', 'High Stress'], filled=True, fontsize=10)
plt.title("Mô Hình Cây Quyết Định (Decision Tree)")
plt.show()
importances = dtModel.feature_importances_
forestImportances = pd.Series(importances, index=features).sort_values(ascending=True)
plt.figure(figsize=(10, 6))
forestImportances.plot(kind='barh', color='teal')
plt.title("Yếu Tố Nào Ảnh Hưởng Nhất Đến Stress?")
plt.xlabel("Mức Độ Quan Trọng (Importance Score)")
plt.ylabel("Các Yếu Tố Hành Vi")
plt.tight_layout()
plt.show()