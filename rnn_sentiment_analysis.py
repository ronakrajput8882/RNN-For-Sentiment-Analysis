import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset

data = pd.read_csv("IMDB Dataset.csv")

data.drop_duplicates(inplace=True)

data["review"] = data["review"].str.lower()

# nltk.download("punkt")
# nltk.download("punkt_tab")
# nltk.download("stopwords")

def remove_urls(text):
    text = re.sub(r"http\S+","",text)
    return text
data["review"] = data["review"].apply(remove_urls)

def remove_punctuations(text):
    text = re.sub(r"[^A-Za-z0-9\s]","",text)
    return text
data["review"] = data["review"].apply(remove_punctuations)

def remove_html(text):
    text = re.sub(r"<.*?>","",text)
    return text
data["review"] = data["review"].apply(remove_html)

stop_words = set(stopwords.words("english"))

def remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if w.lower() not in stop_words]
    return " ".join(filtered)
data["review"] = data["review"].apply(remove_stopwords)    

def stemming(text):
    ps = PorterStemmer()
    stemmed_words = []
    
    tokens = word_tokenize(text)
    for token in tokens:
        stemmed_token = ps.stem(token)
        stemmed_words.append(stemmed_token)
        
    return " ".join(stemmed_words)

data["review"] = data["review"].apply(stemming)  

le = LabelEncoder()
data["sentiment"] = le.fit_transform(data["sentiment"])
y = data["sentiment"]

tf = TfidfVectorizer(max_features=5000)
X = tf.fit_transform(data["review"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = X_train.toarray()
X_test = X_test.toarray()

train_set = TensorDataset(
    torch.from_numpy(X_train).float(),
    torch.from_numpy(y_train.values).float(),
)

test_set = TensorDataset(
    torch.from_numpy(X_test).float(),
    torch.from_numpy(y_test.values).float(),
)

Train_loader = DataLoader(train_set,batch_size=64,shuffle=True)
test_loader = DataLoader(test_set,batch_size=64,shuffle=True)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layes = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size,1)
    def forward(self,x):
        h0 = torch.zeros(self.num_layes, x.size(0), self.hidden_size)
        
        out, _ = self.rnn(x,h0)
        
        out = self.fc(out[:, -1, :])
        return out

input_size = X_train.shape[1]

model = RNN(input_size)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

epochs = 10

for epoch in range(epochs):
    model.train()
    
    for Xb, yb in Train_loader:
        optimizer.zero_grad()
        
        Xb = Xb.unsqueeze(1)
        
        outputs = model(Xb)
        outputs = torch.sigmoid(outputs.squeeze())
        
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
    
    print(f"epoch = {epoch + 1}/ {epochs} and the loss = {loss.item()} ")
    

model.eval()

with torch.no_grad():
    correct_vals = 0
    total_vals = 0
    
    for Xb, yb in test_loader:
        Xb = Xb.unsqueeze(1)
        
        outputs = model(Xb)
        pridicted = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
        
        total_vals += yb.size(0)
        correct_vals += (pridicted == yb).sum().item()
    
    print(f"Accuracy = {correct_vals/total_vals*100}")