import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

df = pd.read_csv('Data/labeled_data.csv')

y = df.loc[:, 'class']
X = df.drop(['class'], axis=1)

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=12)

X_train = X_train["tweet"]
X_test = X_test["tweet"]

# Definicja klasyfikatora SVM z wykorzystaniem TF-IDF
classifier = make_pipeline(TfidfVectorizer(), SVC(kernel='linear'))

# Trenowanie klasyfikatora
classifier.fit(X_train, y_train)

# Predykcja na zbiorze testowym
y_pred = classifier.predict(X_test)

# Ocena wydajności klasyfikatora
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:")
print(report)
