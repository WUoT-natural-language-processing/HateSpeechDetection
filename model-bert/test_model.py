import torch
import pandas as pd
from transformers import DistilBertTokenizer
from main import DistilBertClassification

def read_dataset(filename: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(filename)
    y = df.loc[:, 'class']
    X = df.drop(['class'], axis=1)
    return X, y

model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

X_test, y_test = read_dataset("Data/test_data.csv")
X_test = X_test["tweet"].values.tolist()
y_test = y_test.values.tolist()

model_reloaded = DistilBertClassification()
model_reloaded.load_state_dict(torch.load('PyModel.sd'))

prediction = model_reloaded(tokenizer(X_test, padding=True, truncation=True, return_tensors='pt')['input_ids'])
prediction_index = prediction.argmax(axis=1)
accuracy = (prediction_index==torch.tensor(y_test))
test_accuracy = (sum(accuracy) / len(accuracy)).item()

print(f'Test accuracy: {test_accuracy} %')