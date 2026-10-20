from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.linear_model import LogisticRegression
from nltk.stem import PorterStemmer, WordNetLemmatizer # type: ignore

vectorized = CountVectorizer()

data = pd.read_csv("H17/train.csv")

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess_text(text):
    words = str(text).lower().split()
    words = [lemmatizer.lemmatize(w) for w in words]
    words = [stemmer.stem(w) for w in words]
    return " ".join(words)

data["text"] = data["text"].apply(preprocess_text)

target = data["target"]
data = data.drop("target", axis=1)
vectorized.fit(data["text"])
data = vectorized.transform(data["text"])

model = LogisticRegression()
model.fit(data, target)

test_data = pd.read_csv("H17/test.csv")
test_data["text"] = test_data["text"].apply(preprocess_text)
test_data = vectorized.transform(test_data["text"])

predictions = model.predict(test_data)

submission = pd.read_csv("H17/sample_submission.csv")
submission["target"] = predictions
submission.to_csv("H17/submission.csv", index=False)
print(submission.head())