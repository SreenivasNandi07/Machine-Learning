

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

texts = [
    "I love this",
    "This is amazing",
    "I hate this",
    "Worst experience"
]

labels = [1, 1, 0, 0]

# Convert text into vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train model
model = LogisticRegression()
model.fit(X, labels)

# Predict
test = vectorizer.transform(["I really love this product"])
prediction = model.predict(test)

print(prediction)