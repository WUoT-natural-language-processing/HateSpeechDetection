import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from transformers import DistilBertTokenizer, DistilBertModel
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

def read_dataset(filename: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(filename)
    y = df.loc[:, 'class']
    X = df.drop(['class'], axis=1)
    return X, y

def save_data_to_csv(data, path):
    f = open(path, mode='w')
    data.to_csv(f, index=False)
    f.close()

LABELS_NUMBER = 2
EPOCHS = 15
BATCH_SIZE = 32
OPTIMIZER_LEARNING_RATE=5e-5

# Choose pretrained BERT model
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model_db = DistilBertModel.from_pretrained(model_name, num_labels=LABELS_NUMBER)

class DistilBertClassification(nn.Module):
    def __init__(self):
        super(DistilBertClassification, self).__init__()
        self.dbert = model_db
        self.dropout = nn.Dropout(p=0.2)
        self.linear1 = nn.Linear(768,64)
        self.ReLu = nn.ReLU()
        self.linear2 = nn.Linear(64,LABELS_NUMBER)

    def forward(self, x):
        x = self.dbert(input_ids=x)
        x = x["last_hidden_state"][:,0,:]
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.ReLu(x)
        logits = self.linear2(x)
        return logits

# Read data
X_train, y_train = read_dataset("Data/train_data.csv")
X_test, y_test = read_dataset("Data/test_data.csv")

X_train, X_valid, y_train, y_valid =  train_test_split(X_train["tweet"].values.tolist(), y_train.values.tolist(), test_size=0.11)

X_test = X_test["tweet"].values.tolist()
y_test = y_test.values.tolist()

# Model configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DistilBertClassification().to(device)
for param in model.dbert.parameters():
    param.requires_grad = False

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=OPTIMIZER_LEARNING_RATE)

total_params = sum(p.numel() for p in model.parameters())
total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of parameters: ", total_params)
print("Number of trainable parameters: ", total_params_trainable)

# Train
train_dataset = TensorDataset(tokenizer(X_train, padding=True, truncation=True, return_tensors='pt')['input_ids'], torch.tensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = TensorDataset(tokenizer(X_valid, padding=True, truncation=True, return_tensors='pt')['input_ids'], torch.tensor(y_valid))
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

## Prepare plots

history = {}
history["epoch"] = []
history["train_loss"] = []
history["valid_loss"] = []
history["train_accuracy"] = []
history["valid_accuracy"] = []

plt.ion()
fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 5))

ax_acc.set_ylim([0, 1])

ax_loss.set_xlim([1, EPOCHS])
ax_acc.set_xlim([1, EPOCHS])

ax_loss.set_xticks(torch.linspace(1, EPOCHS, EPOCHS))
ax_acc.set_xticks(torch.linspace(1, EPOCHS, EPOCHS))

ax_loss.set_xlabel("Epochs")
ax_loss.set_ylabel("Loss")
ax_acc.set_xlabel("Epochs")
ax_acc.set_ylabel("Accuracy")

line_loss_train, = ax_loss.plot([], [], 'o-')
line_loss_val, = ax_loss.plot([], [], 'o-', color='orange')

line_loss_train.set_label("Train")
line_loss_val.set_label("Validation")
ax_loss.legend(loc='upper left')

line_acc_train, = ax_acc.plot([], [], 'o-')
line_acc_val, = ax_acc.plot([], [], 'o-', color='orange')

line_acc_train.set_label("Train")
line_acc_val.set_label("Validation")
ax_acc.legend(loc='upper left')

plt.show()

start_time = datetime.now()

for e in range(EPOCHS):
    model.train()

    train_loss = 0.0
    train_accuracy = []

    for X, y in tqdm(train_loader):
        # Get prediction and loss
        prediction = model(X)
        loss = criterion(prediction, y) 

        # Adjust the parameters of the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        
        prediction_index = prediction.argmax(axis=1)
        accuracy = (prediction_index==y)
        train_accuracy += accuracy

    train_accuracy = (sum(train_accuracy) / len(train_accuracy)).item()

    # Evaluate model
    model.eval()

    valid_loss = 0.0
    valid_accuracy = []
    
    for X, y in tqdm(val_loader):
        prediction = model(X)
        loss = criterion(prediction, y)

        valid_loss += loss.item()
        
        prediction_index = prediction.argmax(axis=1)
        accuracy = (prediction_index==y)
        valid_accuracy += accuracy
    
    valid_accuracy = (sum(valid_accuracy) / len(valid_accuracy)).item()

    # Populate history
    history["epoch"].append(e+1)
    history["train_loss"].append(train_loss / len(train_loader))
    history["valid_loss"].append(valid_loss / len(val_loader))
    history["train_accuracy"].append(train_accuracy)
    history["valid_accuracy"].append(valid_accuracy)

    # Update plots
    ax_loss.set_ylim([min(min(history["valid_loss"], history["train_loss"])) * 0.75, max(max(history["valid_loss"], history["train_loss"])) * 1.25])

    line_loss_train.set_data(history["epoch"], history["train_loss"])
    line_loss_val.set_data(history["epoch"], history["valid_loss"])

    line_acc_train.set_data(history["epoch"], history["train_accuracy"])
    line_acc_val.set_data(history["epoch"], history["valid_accuracy"])

    plt.pause(0.1)

# Dump model
torch.save(model.state_dict(), 'PyModel.sd')

plt.savefig('model_result.png')

# Measure time for training
end_time = datetime.now()
training_time = (end_time - start_time).total_seconds() / 60

best = torch.argmax(torch.tensor(history["valid_accuracy"])).item()
print(f'Training time: {training_time} min')
print()
print(f'Validation accuracy: {history["valid_accuracy"][best] * 100} %')
print(f'Validation loss: {history["valid_loss"][best]}')
print()
