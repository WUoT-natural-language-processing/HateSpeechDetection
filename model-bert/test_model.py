import torch
import numpy
from transformers import DistilBertTokenizer
from main import MODEL_NAME, DistilBertClassification, read_dataset

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

X_test, y_test = read_dataset("Data/test_data.csv")
X_test = X_test["tweet"].values[:100].tolist()
y_test = y_test.values[:100].tolist()

model_reloaded = DistilBertClassification()
model_reloaded.load_state_dict(torch.load('results/lr_5e-5_v2.sd'))

model_reloaded.eval()
prediction = model_reloaded(tokenizer(X_test, padding=True, truncation=True, return_tensors='pt')['input_ids'])
prediction_index = prediction.argmax(axis=1)
matches = (prediction_index==torch.tensor(y_test))

indices = numpy.nonzero(1 - matches.numpy())[0]
X_wrong = numpy.array(X_test)[indices.astype(int)]
y_wrong = numpy.array(y_test)[indices.astype(int)]

test_accuracy = (sum(matches) / len(matches)).item()

print(f'Test accuracy: {test_accuracy * 100} %')