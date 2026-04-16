import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import joblib
import sys
import io
import re
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score, classification_report, f1_score, precision_score, recall_score)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-whitegrid')
filePath = "NCKH.csv"
df = pd.read_csv(filePath)
df.columns = df.columns.str.strip()
colMap = {
  'Trung bình bạn mất bao lâu để bắt đầu làm bài tập sau khi được giao (tính bằng giờ)?': 'responseTime',
  'Trong 4 tuần gần nhất, bạn nộp muộn bao nhiêu lần?': 'lateCount',
  'Bạn truy cập hệ thống học tập (LMS, portal lớp) trung bình bao nhiêu lần mỗi ngày?': 'lmsAccess',
  'Thời điểm bạn thường truy cập nhất:': 'accessTiming',
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
  vals = [float(n.replace(',', '.')) for n in nums]
  val = sum(vals) / len(vals)
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
negWords = ['căng thẳng', 'mệt', 'áp lực', 'lo lắng', 'sợ', 'chán', 'khó', 'stress', 'buồn', 'tệ', 'kém', 'thất bại', 'không thể', 'quá tải', 'kiệt sức', 'không hiểu', 'khó khăn', 'nặng nề', 'mất ngủ', 'lo âu']
posWords = ['vui', 'tốt', 'ổn', 'thoải mái', 'tự tin', 'thích', 'hứng thú', 'hiểu', 'dễ', 'hài lòng', 'tích cực', 'yêu thích', 'hứng khởi']
def sentimentScore(text):
  if pd.isna(text): return 0
  text = str(text).lower()
  neg = sum(1 for w in negWords if w in text)
  pos = sum(1 for w in posWords if w in text)
  return neg - pos
ordinalMaps = {
  'sleepQuality': {'rất tệ': 1, 'tệ': 2, 'kém': 2, 'trung bình': 3, 'bình thường': 3, 'tốt': 4, 'rất tốt': 5},
  'interactionScore': {'rất thấp': 1, 'thấp': 2, 'ít': 2, 'trung bình': 3, 'cao': 4, 'thường xuyên': 4, 'rất cao': 5},
  'procrastinationLevel': {'không bao giờ': 1, 'hiếm': 2, 'đôi khi': 3, 'thỉnh thoảng': 3, 'thường xuyên': 4, 'luôn luôn': 5},
  'accessTiming': {'sáng': 1, 'trưa': 1, 'chiều': 1, 'tối': 2, 'khuya': 3}
}
def cleanOrdinal(text, mapping):
  if pd.isna(text): return np.nan
  text = str(text).strip().lower()
  for key, val in mapping.items():
    if key in text: return val
  nums = re.findall(r'\d+', text)
  return float(nums[0]) if nums else np.nan
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
for col, mapping in ordinalMaps.items():
  dfClean[col] = dfClean[col].apply(lambda x: cleanOrdinal(x, mapping))
  dfClean[col] = pd.to_numeric(dfClean[col], errors='coerce')
  dfClean[col] = dfClean[col].fillna(dfClean[col].median())
dfClean['sentimentIntensity'] = dfClean['emotionalText'].apply(sentimentScore)
dfClean['healthIndex'] = dfClean['sleepHours'] * dfClean['sleepQuality']
dfClean['digitalInteraction'] = dfClean['lmsAccess'] * dfClean['interactionScore']
features = ['responseTime', 'lateCount', 'lmsAccess', 'accessTiming', 'sentimentIntensity', 'sleepHours', 'procrastinationLevel', 'healthIndex', 'digitalInteraction']
target = 'stressScore'
feat_map = {f: re.sub(r'([a-z])([A-Z])', r'\1 \2', f).title() for f in features + [target]}
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
print("Model Training, Ensemble & Comparison")
print("=" * 80)
print(f"\nDataset Size: {len(dfFinal)} Samples")
print(f"Training Set: {len(xTrainReg)} Samples | Test Set: {len(xTestReg)} Samples")
print(f"Class Distribution: Normal={sum(yClass == 0)} | High Stress={sum(yClass == 1)}")
print(f"Feature Count: {len(features)}")
print("\n" + "=" * 80)
print("Model 1: Linear Regression (Baseline For Regression)")
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
print("Model 2: Decision Tree Classifier (Baseline For Classification)")
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
print(classification_report(yTestClass, yPredDtTest, target_names=['Normal', 'High Stress']))
print("\n" + "=" * 80)
print("Phase 5: Handling Class Imbalance With SMOTE")
print("=" * 80)
smote = SMOTE(random_state=42)
xTrainSmote, yTrainSmote = smote.fit_resample(xTrainClass, yTrainClass)
print(f"\nClass Distribution After SMOTE:")
print(f"  - Original: Normal={sum(yTrainClass == 0)} | High Stress={sum(yTrainClass == 1)}")
print(f"  - After SMOTE: Normal={sum(yTrainSmote == 0)} | High Stress={sum(yTrainSmote == 1)}")
print("\n" + "=" * 80)
print("Model 3: Random Forest Classifier (Ensemble Method)")
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
print("Model 4: Gradient Boosting Classifier (Advanced Ensemble)")
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
print("Model 5: Voting Classifier (Optimal Ensemble Combination)")
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
print("Comprehensive Model Comparison")
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
print("Cross-Validation Scores (5-Fold)")
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
print("Feature Importance Analysis")
print("=" * 80)
print("\nRandom Forest Feature Importance:")
rfImportances = pd.Series(rfModel.feature_importances_, index=[feat_map[f] for f in features]).sort_values(ascending=False)
print(rfImportances)
print("\nGradient Boosting Feature Importance:")
gbImportances = pd.Series(gbModel.feature_importances_, index=[feat_map[f] for f in features]).sort_values(ascending=False)
print(gbImportances)
print("\n" + "=" * 80)
print("Generating Visualizations...")
print("=" * 80)
plt.figure(figsize=(12, 10))
sns.heatmap(dfFinal[features + [target]].rename(columns=feat_map).corr(), annot=True, cmap='coolwarm', fmt=".2f", square=True,
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
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
dtImportances = pd.Series(dtModel.feature_importances_, index=[feat_map[f] for f in features]).sort_values()
dtImportances.plot(kind='barh', ax=axes[0, 0], color='lightseagreen')
axes[0, 0].set_title('Decision Tree Importance', fontsize=16, fontweight='bold', pad=20)
rfImportancesFormatted = pd.Series(rfModel.feature_importances_, index=[feat_map[f] for f in features]).sort_values()
rfImportancesFormatted.plot(kind='barh', ax=axes[0, 1], color='steelblue')
axes[0, 1].set_title('Random Forest Importance', fontsize=16, fontweight='bold', pad=20)
gbImportancesFormatted = pd.Series(gbModel.feature_importances_, index=[feat_map[f] for f in features]).sort_values()
gbImportancesFormatted.plot(kind='barh', ax=axes[1, 0], color='coral')
axes[1, 0].set_title('Gradient Boosting Importance', fontsize=16, fontweight='bold', pad=20)
avgVotingImportance = (dtModel.feature_importances_ + rfModel.feature_importances_ + gbModel.feature_importances_) / 3
voteImportances = pd.Series(avgVotingImportance, index=[feat_map[f] for f in features]).sort_values()
voteImportances.plot(kind='barh', ax=axes[1, 1], color='mediumpurple')
axes[1, 1].set_title('Voting Classifier Importance', fontsize=16, fontweight='bold', pad=20)
for ax in axes.flat:
  ax.set_xlabel('Importance Score')
plt.subplots_adjust(hspace=0.25, wspace=0.3, top=0.88, bottom=0.08)
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
plt.figure(figsize=(80, 20))
plot_tree(dtModel, feature_names=[feat_map[f] for f in features], class_names=['Bình Thường', 'Stress Cao'], filled=True, rounded=True, precision=1, fontsize=8, impurity=False)
plt.title("Decision Tree Visualization", fontsize=40, fontweight='bold')
plt.subplots_adjust(left=0.01, right=0.99, top=0.75, bottom=0.05)
plt.savefig('DecisionTree.png', dpi=300)
plt.show()
avgImportances = (rfImportances + gbImportances) / 2
plt.figure(figsize=(10, 6))
avgImportances.sort_values().plot(kind='barh', color='purple')
plt.title('Average Feature Importance Score', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('ImportanceScore.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n" + "=" * 80)
print("All Visualizations Saved Successfully!")
print("=" * 80)
print("\n" + "=" * 80)
print("Final Recommendation")
print("=" * 80)
bestModel = max([('Decision Tree', accDtTest), ('Random Forest', accRfTest),
                 ('Gradient Boosting', accGbTest), ('Voting Classifier', accVoteTest)],
                key=lambda x: x[1])
print(f"\n✓ Best Overall Model: {bestModel[0]}")
print(f"  - Test Accuracy: {bestModel[1]:.4f}")
model = {'Decision Tree': dtModel, 'Random Forest': rfModel, 'Gradient Boosting': gbModel, 'Voting Classifier': votingModel}
joblib.dump(model[bestModel[0]], 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print(f"✓ Model & Scaler Exported Successfully To 'model.pkl' & 'scaler.pkl'")
print(f"\n✓ Recommendation: Use {bestModel[0]} For Production Deployment")
print(f"✓ Key Improvements Applied:")
print(f"  1. SMOTE For Class Imbalance Handling")
print(f"  2. Ensemble Methods (Random Forest, Gradient Boosting, Voting)")
print(f"  3. Cross-Validation For Robust Evaluation")
print(f"  4. Comprehensive Metrics Beyond Accuracy")
print(f"  5. Feature Importance Analysis For Interpretability")
print("\n" + "=" * 80)