import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('malicious_phish.csv')

# Encode labels: benign = 0, all others = 1 (malicious)
df['label'] = df['type'].apply(lambda x: 0 if x == 'benign' else 1)

# View class balance
print("Label Distribution:")
print(df['label'].value_counts())

# Features and labels
X = df['url']
y = df['label']

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train balanced model
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model and vectorizer
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved successfully.")

# Graph: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Safe', 'Malicious'], yticklabels=['Safe', 'Malicious'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# Graph: Classification Report
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose().iloc[:2, :3]  # class 0 and 1 only

report_df.plot(kind='bar', figsize=(7, 4), ylim=(0, 1.1), colormap='Set2')
plt.title("Classification Report: Precision, Recall, F1-score")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
