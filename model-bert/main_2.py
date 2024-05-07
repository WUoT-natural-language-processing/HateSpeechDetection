import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

def read_dataset(filename: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(filename)
    y = df.loc[:, 'class']
    X = df.drop(['class'], axis=1)
    return X, y

def save_data_to_csv(data, path):
    f = open(path, mode='w')
    data.to_csv(f, index=False)
    f.close()
    
# read data
X_train, y_train = read_dataset("Data/train_data.csv")
X_test, y_test = read_dataset("Data/test_data.csv")

X_train = X_train["tweet"].values.tolist()
X_test = X_test["tweet"].values.tolist()

# choose pretrained BERT model
model_name = 'distilbert/distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, trainable=False)

# configure torch device and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# train
batch_size = 32

train_dataset = TensorDataset(tokenizer(X_train, padding=True, truncation=True, return_tensors='pt')['input_ids'], torch.tensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(tokenizer(X_test, padding=True, truncation=True, return_tensors='pt')['input_ids'], torch.tensor(y_test))
val_loader = DataLoader(val_dataset, batch_size=batch_size)

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

epochs = 5
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    correct_train = 0
    total_train = 0

    for batch in train_loader:
        input_ids, labels = batch
        input_ids, labels = input_ids.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        _, predicted = torch.max(outputs.logits, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    train_loss = total_train_loss / len(train_loader)
    train_accuracy = correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # evaluate model
    model.eval()
    total_val_loss = 0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids, labels = batch
            input_ids, labels = input_ids.to(device), labels.to(device)
            
            outputs = model(input_ids)
            loss = outputs.loss
            
            total_val_loss += loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)
    
    val_loss = total_val_loss / len(val_loader)
    val_accuracy = correct_val / total_val
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()