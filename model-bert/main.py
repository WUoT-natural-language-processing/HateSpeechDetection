import pandas as pd
import torch
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
EPOCHS = 10
BATCH_SIZE = 32
OPTIMIZER_LEARNING_RATE=1e-4

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

class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(
        self, optimizer, patience=2, min_lr=1e-6, factor=0.5
    ):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
            )
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)

# Read data
X_train, y_train = read_dataset("Data/train_data.csv")
X_test, y_test = read_dataset("Data/test_data.csv")

X_train = X_train["tweet"].values.tolist()
y_train = y_train
X_test = X_test["tweet"].values.tolist()
y_test = y_test

# Model configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DistilBertClassification().to(device)
for param in model.dbert.parameters():
    param.requires_grad = False

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=OPTIMIZER_LEARNING_RATE)
lr_scheduler = LRScheduler(optimizer)

total_params = sum(p.numel() for p in model.parameters())
total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of parameters: ", total_params)
print("Number of trainable parameters: ", total_params_trainable)

# Train
train_dataset = TensorDataset(tokenizer(X_train, padding=True, truncation=True, return_tensors='pt')['input_ids'], torch.tensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = TensorDataset(tokenizer(X_test, padding=True, truncation=True, return_tensors='pt')['input_ids'], torch.tensor(y_test))
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

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

    # Update learning rate
    lr_scheduler(history["valid_loss"][-1])

    # Update plots
    ax_loss.set_ylim([min(min(history["valid_loss"], history["train_loss"])) * 0.75, max(max(history["valid_loss"], history["train_loss"])) * 1.25])

    line_loss_train.set_data(history["epoch"], history["train_loss"])
    line_loss_val.set_data(history["epoch"], history["valid_loss"])

    line_acc_train.set_data(history["epoch"], history["train_accuracy"])
    line_acc_val.set_data(history["epoch"], history["valid_accuracy"])

    plt.pause(0.1)
    
# Measure time for training
end_time = datetime.now()
training_time = (end_time - start_time).total_seconds() / 60

print(f'Validation accuracy: {history["valid_accuracy"][-1] * 100} %')
print(f'Validation loss: {history["valid_loss"][-1]}')
print(f'Training time: {training_time} min')

torch.save(model.state_dict(), 'PyModel.sd')

plt.savefig('model_result.png')