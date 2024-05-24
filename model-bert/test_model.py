import matplotlib.pyplot as plt
import numpy
import torch
from transformers import DistilBertTokenizer

from main import MODEL_NAME, TEST_DATA_PATH, DistilBertClassification, read_dataset

MODEL_PATH = "results/lr_5e-5_v2.sd"

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

# Read test data
X_test, y_test = read_dataset(TEST_DATA_PATH)
X_test = X_test["tweet"].values.tolist()
y_test = y_test.values.tolist()

# Load model
model_reloaded = DistilBertClassification()
model_reloaded.load_state_dict(torch.load(MODEL_PATH))

# Test model
model_reloaded.eval()
prediction = model_reloaded(tokenizer(
    X_test, padding=True, truncation=True, return_tensors='pt')['input_ids'])
prediction_index = prediction.argmax(axis=1)
matches = prediction_index == torch.tensor(y_test)
test_accuracy = (sum(matches) / len(matches)).item()

print(f'Test accuracy: {test_accuracy * 100} %')

# Incorrect predictions analysis
indices = numpy.nonzero(1 - matches.numpy())[0]
X_wrong = numpy.array(X_test)[indices.astype(int)]
y_wrong = numpy.array(y_test)[indices.astype(int)]

incorrect = numpy.array(X_test)[indices.astype(int)]
incorrect_lens = [len(i) for i in incorrect]

print(incorrect)

plt.hist(incorrect_lens, bins=10, color='skyblue', edgecolor='black')
plt.xlabel('Długość tweeta')
plt.ylabel('Liczba tweetów')
plt.title('Histogram długości tweetów sklasyfikowanych błędnie\nprzez model zbudowany z użyciem BERTa')
plt.savefig('model-bert/hist-incorrect-bert.png')
plt.show()
