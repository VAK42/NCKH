import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score, classification_report, f1_score, precision_score, recall_score)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')
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
x = dfFinal[features]
y = dfFinal[target]
scaler = StandardScaler()
xScaled = scaler.fit_transform(x)
xTrainReg, xTestReg, yTrainReg, yTestReg = train_test_split(
  xScaled, y, test_size=0.2, random_state=42
)
yClass = (y >= 4).astype(int)
xTrainClass, xTestClass, yTrainClass, yTestClass = train_test_split(
  xScaled, yClass, test_size=0.2, random_state=42
)
print("=" * 80)
print("MODEL TRAINING, ENSEMBLE & COMPARISON")
print("=" * 80)
print(f"\nDataset Size: {len(dfFinal)} Samples")
print(f"Training Set: {len(xTrainReg)} Samples | Test Set: {len(xTestReg)} Samples")
print(f"Class Distribution: Normal={sum(yClass == 0)} | High Stress={sum(yClass == 1)}")
print(f"Feature Count: {len(features)}")
print("\n" + "=" * 80)
print("MODEL 1: LINEAR REGRESSION (BASELINE FOR REGRESSION)")
print("=" * 80)
linReg = LinearRegression()
linReg.fit(xTrainReg, yTrainReg)
yPredLinTrain = linReg.predict(xTrainReg)
yPredLinTest = linReg.predict(xTestReg)
mseLinTrain = mean_squared_error(yTrainReg, yPredLinTrain)
mseLinTest = mean_squared_error(yTestReg, yPredLinTest)
r2LinTrain = r2_score(yTrainReg, yPredLinTrain)
r2LinTest = r2_score(yTestReg, yPredLinTest)
print(f"\nTraining Metrics:")
print(f"  - MSE: {mseLinTrain:.4f}")
print(f"  - R² Score: {r2LinTrain:.4f}")
print(f"\nTest Metrics:")
print(f"  - MSE: {mseLinTest:.4f}")
print(f"  - R² Score: {r2LinTest:.4f}")
print("\n" + "=" * 80)
print("MODEL 2: DECISION TREE CLASSIFIER (BASELINE FOR CLASSIFICATION)")
print("=" * 80)
dtModel = DecisionTreeClassifier(max_depth=4, random_state=42)
dtModel.fit(xTrainClass, yTrainClass)
yPredDtTrain = dtModel.predict(xTrainClass)
yPredDtTest = dtModel.predict(xTestClass)
accDtTrain = accuracy_score(yTrainClass, yPredDtTrain)
accDtTest = accuracy_score(yTestClass, yPredDtTest)
print(f"\nTraining Metrics:")
print(f"  - Accuracy: {accDtTrain:.4f}")
print(f"  - Precision: {precision_score(yTrainClass, yPredDtTrain):.4f}")
print(f"  - Recall: {recall_score(yTrainClass, yPredDtTrain):.4f}")
print(f"  - F1-Score: {f1_score(yTrainClass, yPredDtTrain):.4f}")
print(f"\nTest Metrics:")
print(f"  - Accuracy: {accDtTest:.4f}")
print(f"  - Precision: {precision_score(yTestClass, yPredDtTest):.4f}")
print(f"  - Recall: {recall_score(yTestClass, yPredDtTest):.4f}")
print(f"  - F1-Score: {f1_score(yTestClass, yPredDtTest):.4f}")
print(f"\nClassification Report (Test Set):")
print(classification_report(yTestClass, yPredDtTest, target_names=['Bình Thường', 'Stress Cao']))
print("\n" + "=" * 80)
print("PHASE 5: HANDLING CLASS IMBALANCE WITH SMOTE")
print("=" * 80)
smote = SMOTE(random_state=42)
xTrainSmote, yTrainSmote = smote.fit_resample(xTrainClass, yTrainClass)
print(f"\nClass Distribution After SMOTE:")
print(f"  - Original: Normal={sum(yTrainClass == 0)} | High Stress={sum(yTrainClass == 1)}")
print(f"  - After SMOTE: Normal={sum(yTrainSmote == 0)} | High Stress={sum(yTrainSmote == 1)}")
print("\n" + "=" * 80)
print("MODEL 3: RANDOM FOREST CLASSIFIER (ENSEMBLE METHOD)")
print("=" * 80)
rfModel = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
rfModel.fit(xTrainSmote, yTrainSmote)
yPredRfTrain = rfModel.predict(xTrainClass)
yPredRfTest = rfModel.predict(xTestClass)
accRfTrain = accuracy_score(yTrainClass, yPredRfTrain)
accRfTest = accuracy_score(yTestClass, yPredRfTest)
print(f"\nTraining Metrics:")
print(f"  - Accuracy: {accRfTrain:.4f}")
print(f"  - Precision: {precision_score(yTrainClass, yPredRfTrain):.4f}")
print(f"  - Recall: {recall_score(yTrainClass, yPredRfTrain):.4f}")
print(f"  - F1-Score: {f1_score(yTrainClass, yPredRfTrain):.4f}")
print(f"\nTest Metrics:")
print(f"  - Accuracy: {accRfTest:.4f}")
print(f"  - Precision: {precision_score(yTestClass, yPredRfTest):.4f}")
print(f"  - Recall: {recall_score(yTestClass, yPredRfTest):.4f}")
print(f"  - F1-Score: {f1_score(yTestClass, yPredRfTest):.4f}")
print("\n" + "=" * 80)
print("MODEL 4: GRADIENT BOOSTING CLASSIFIER (ADVANCED ENSEMBLE)")
print("=" * 80)
gbModel = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbModel.fit(xTrainSmote, yTrainSmote)
yPredGbTrain = gbModel.predict(xTrainClass)
yPredGbTest = gbModel.predict(xTestClass)
accGbTrain = accuracy_score(yTrainClass, yPredGbTrain)
accGbTest = accuracy_score(yTestClass, yPredGbTest)
print(f"\nTraining Metrics:")
print(f"  - Accuracy: {accGbTrain:.4f}")
print(f"  - Precision: {precision_score(yTrainClass, yPredGbTrain):.4f}")
print(f"  - Recall: {recall_score(yTrainClass, yPredGbTrain):.4f}")
print(f"  - F1-Score: {f1_score(yTrainClass, yPredGbTrain):.4f}")
print(f"\nTest Metrics:")
print(f"  - Accuracy: {accGbTest:.4f}")
print(f"  - Precision: {precision_score(yTestClass, yPredGbTest):.4f}")
print(f"  - Recall: {recall_score(yTestClass, yPredGbTest):.4f}")
print(f"  - F1-Score: {f1_score(yTestClass, yPredGbTest):.4f}")
print("\n" + "=" * 80)
print("MODEL 5: VOTING CLASSIFIER (OPTIMAL ENSEMBLE COMBINATION)")
print("=" * 80)
votingModel = VotingClassifier(
  estimators=[
    ('dt', DecisionTreeClassifier(max_depth=4, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)),
    ('lr', LogisticRegression(max_iter=1000, random_state=42))
  ],
  voting='soft'
)
votingModel.fit(xTrainSmote, yTrainSmote)
yPredVoteTrain = votingModel.predict(xTrainClass)
yPredVoteTest = votingModel.predict(xTestClass)
accVoteTrain = accuracy_score(yTrainClass, yPredVoteTrain)
accVoteTest = accuracy_score(yTestClass, yPredVoteTest)
print(f"\nTraining Metrics:")
print(f"  - Accuracy: {accVoteTrain:.4f}")
print(f"  - Precision: {precision_score(yTrainClass, yPredVoteTrain):.4f}")
print(f"  - Recall: {recall_score(yTrainClass, yPredVoteTrain):.4f}")
print(f"  - F1-Score: {f1_score(yTrainClass, yPredVoteTrain):.4f}")
print(f"\nTest Metrics:")
print(f"  - Accuracy: {accVoteTest:.4f}")
print(f"  - Precision: {precision_score(yTestClass, yPredVoteTest):.4f}")
print(f"  - Recall: {recall_score(yTestClass, yPredVoteTest):.4f}")
print(f"  - F1-Score: {f1_score(yTestClass, yPredVoteTest):.4f}")
print("\n" + "=" * 80)
print("COMPREHENSIVE MODEL COMPARISON")
print("=" * 80)
comparisonData = {
  'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'Voting Classifier'],
  'Test Accuracy': ['-', accDtTest, accRfTest, accGbTest, accVoteTest],
  'Test Precision': ['-', precision_score(yTestClass, yPredDtTest),
                     precision_score(yTestClass, yPredRfTest),
                     precision_score(yTestClass, yPredGbTest),
                     precision_score(yTestClass, yPredVoteTest)],
  'Test Recall': ['-', recall_score(yTestClass, yPredDtTest),
                  recall_score(yTestClass, yPredRfTest),
                  recall_score(yTestClass, yPredGbTest),
                  recall_score(yTestClass, yPredVoteTest)],
  'Test F1-Score': ['-', f1_score(yTestClass, yPredDtTest),
                    f1_score(yTestClass, yPredRfTest),
                    f1_score(yTestClass, yPredGbTest),
                    f1_score(yTestClass, yPredVoteTest)]
}
comparisonDf = pd.DataFrame(comparisonData)
print("\n", comparisonDf.to_string(index=False))
print("\n" + "=" * 80)
print("CROSS-VALIDATION SCORES (5-FOLD)")
print("=" * 80)
cvModels = {
  'Decision Tree': DecisionTreeClassifier(max_depth=4, random_state=42),
  'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
  'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
  'Voting Classifier': votingModel
}
cvScores = {}
for modelName, model in cvModels.items():
  scores = cross_val_score(model, xTrainSmote, yTrainSmote, cv=5, scoring='accuracy', n_jobs=-1)
  cvScores[modelName] = {
    'Mean': scores.mean(),
    'Std': scores.std(),
    'Scores': scores
  }
  print(f"\n{modelName}:")
  print(f"  - Mean Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
  print(f"  - Fold Scores: {[f'{s:.4f}' for s in scores]}")
print("\n" + "=" * 80)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)
print("\nRandom Forest Feature Importance:")
rfImportances = pd.Series(rfModel.feature_importances_, index=features).sort_values(ascending=False)
print(rfImportances)
print("\nGradient Boosting Feature Importance:")
gbImportances = pd.Series(gbModel.feature_importances_, index=features).sort_values(ascending=False)
print(gbImportances)
print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS...")
print("=" * 80)
plt.figure(figsize=(12, 10))
sns.heatmap(dfFinal[features + [target]].corr(), annot=True, cmap='coolwarm', fmt=".2f", square=True,
            cbar_kws={"shrink": 0.8})
plt.title("Ma Trận Tương Quan (Correlation Matrix)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('CorrelationMatrix.png', dpi=300, bbox_inches='tight')
plt.show()
fig, ax = plt.subplots(figsize=(10, 6))
models = ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'Voting Classifier']
accuracies = [accDtTest, accRfTest, accGbTest, accVoteTest]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
bars = ax.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Model Comparison: Test Accuracy', fontsize=14, fontweight='bold')
ax.set_ylim([0, 1.0])
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
  ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f'{acc:.4f}', ha='center', va='bottom',
          fontweight='bold')
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig('ModelAccuracyComparison.png', dpi=300, bbox_inches='tight')
plt.show()
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
rfImportances.sort_values().plot(kind='barh', ax=axes[0], color='steelblue')
axes[0].set_title('Random Forest Feature Importance', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Importance Score')
gbImportances.sort_values().plot(kind='barh', ax=axes[1], color='coral')
axes[1].set_title('Gradient Boosting Feature Importance', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('FeatureImportanceComparison.png', dpi=300, bbox_inches='tight')
plt.show()
fig, ax = plt.subplots(figsize=(12, 6))
metrics = ['Precision', 'Recall', 'F1-Score']
x = np.arange(len(metrics))
width = 0.2
dtVals = [precision_score(yTestClass, yPredDtTest),
          recall_score(yTestClass, yPredDtTest),
          f1_score(yTestClass, yPredDtTest)]
rfVals = [precision_score(yTestClass, yPredRfTest),
          recall_score(yTestClass, yPredRfTest),
          f1_score(yTestClass, yPredRfTest)]
gbVals = [precision_score(yTestClass, yPredGbTest),
          recall_score(yTestClass, yPredGbTest),
          f1_score(yTestClass, yPredGbTest)]
voteVals = [precision_score(yTestClass, yPredVoteTest),
            recall_score(yTestClass, yPredVoteTest),
            f1_score(yTestClass, yPredVoteTest)]
ax.bar(x - 1.5 * width, dtVals, width, label='Decision Tree', color='#FF6B6B')
ax.bar(x - 0.5 * width, rfVals, width, label='Random Forest', color='#4ECDC4')
ax.bar(x + 0.5 * width, gbVals, width, label='Gradient Boosting', color='#45B7D1')
ax.bar(x + 1.5 * width, voteVals, width, label='Voting Classifier', color='#96CEB4')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Model Performance: Precision vs Recall vs F1-Score', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.set_ylim([0, 1.0])
plt.tight_layout()
plt.savefig('PrecisionRecallF1Comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n" + "=" * 80)
print("ALL VISUALIZATIONS SAVED SUCCESSFULLY!")
print("=" * 80)
print("\n" + "=" * 80)
print("FINAL RECOMMENDATION")
print("=" * 80)
bestModel = max([('Decision Tree', accDtTest), ('Random Forest', accRfTest),
                 ('Gradient Boosting', accGbTest), ('Voting Classifier', accVoteTest)],
                key=lambda x: x[1])
print(f"\n✓ BEST OVERALL MODEL: {bestModel[0]}")
print(f"  - Test Accuracy: {bestModel[1]:.4f}")
print(f"\n✓ Recommendation: Use {bestModel[0]} for Production Deployment")
print(f"✓ Key Improvements Applied:")
print(f"  1. SMOTE For Class Imbalance Handling")
print(f"  2. Ensemble Methods (Random Forest, Gradient Boosting, Voting)")
print(f"  3. Cross-Validation For Robust Evaluation")
print(f"  4. Comprehensive Metrics Beyond Accuracy")
print(f"  5. Feature Importance Analysis For Interpretability")
print("\n" + "=" * 80)
