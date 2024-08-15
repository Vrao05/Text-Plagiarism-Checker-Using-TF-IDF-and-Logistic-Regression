from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

text_files = ['file1.txt', 'file2.txt', 'file3.txt', 'file4.txt', 'file5.txt']
texts = []

for file in text_files:
    with open(file, 'r') as f:
        text_data = f.read()
        texts.extend([line.strip() for line in text_data.split('\n') if line.strip()])
df = pd.DataFrame({'text': texts})
training_data, testing_data = train_test_split(df, random_state=2000)
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
X_train = vectorizer.fit_transform(training_data['text'])
y_train = training_data['text']  # assuming the text is the target variable
X_test = vectorizer.transform(testing_data['text'])
y_test = testing_data['text']

logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)

logistic_preds = logistic_model.predict(X_test)
gb_preds = gb_model.predict(X_test)

logistic_accuracy = accuracy_score(y_test, logistic_preds)
logistic_precision = precision_score(y_test, logistic_preds, average='weighted')
logistic_recall = recall_score(y_test, logistic_preds, average='weighted')
logistic_f1 = f1_score(y_test, logistic_preds, average='weighted')

gb_accuracy = accuracy_score(y_test, gb_preds)
gb_precision = precision_score(y_test, gb_preds, average='weighted')
gb_recall = recall_score(y_test, gb_preds, average='weighted')
gb_f1 = f1_score(y_test, gb_preds, average='weighted')

print("Logistic Regression:")
print(f"Training Accuracy: {logistic_model.score(X_train, y_train):.2f}%")
print(f"Testing Accuracy: {logistic_accuracy:.2f}%")
print(f"Precision: {logistic_precision:.2f}%")
print(f"Recall: {logistic_recall:.2f}%")
print(f"F1-Score: {logistic_f1:.2f}%")

print("\nGradient Boosting Classifier:")
print(f"Training Accuracy: {gb_model.score(X_train, y_train):.2f}%")
print(f"Testing Accuracy: {gb_accuracy:.2f}%")
print(f"Precision: {gb_precision:.2f}%")
print(f"Recall: {gb_recall:.2f}%")
print(f"F1-Score: {gb_f1:.2f}%")
print(f"\nAccuracy Comparison: Logistic Regression ({logistic_accuracy:.2f}%) vs Gradient Boosting Classifier ({gb_accuracy:.2f}%)")
