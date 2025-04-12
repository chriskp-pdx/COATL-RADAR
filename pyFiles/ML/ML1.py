# COATL-RADAR ML Bean Detection Program V1
# Chris Kane-Pardy & Kamal Smith

# Import PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Initial Model Prototype
class Model(nn.Module):
    # Input Layer (Bean Amplitude)
    # Hidden Layer 1 & 2 (4 Neurons)
    # Output (Bean Moisture Content (High, Medium, Low))
    def __init__(self, in_features=1, h1=128, h2=64, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
          
# Create Random Seed 
torch.manual_seed(30)

# Instantiate Model
model = Model()

# Load Data
df = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\Git 4 School\COATL-RADAR\pyFiles\Dataset\BeanMaxNormalized_Output_ConstantAmplitudeV4 - ToExport3.csv")
x = df.drop(['Bean Name', 'Bean Number'], axis=1)
y = df['Bean Number']

# Encode Target Labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Maps labels to 0, 1, 2

# Print unique labels to verify
print("Unique labels after encoding:", np.unique(y_encoded))

X = x.values
Y = y_encoded  # Use encoded labels

# Split Data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=30)

# Convert to Tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
Y_train = torch.LongTensor(Y_train)
Y_test = torch.LongTensor(Y_test)

# Set Criterion
criterion = nn.CrossEntropyLoss()

# Choose Adam Optimizer and Learning Rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.000000001)

# Train the Model
epochs = 100000
losses = []
for i in range(epochs):
    # Forward Pass
    y_pred = model(X_train)  # Use model(X_train) instead of model.forward
    # Compute Loss
    loss = criterion(y_pred, Y_train)
    # Store Loss
    losses.append(loss.item())  # Use .item() instead of .detach().numpy()
    # Print Progress
    if i % 10 == 0:
        print(f'Epoch: {i}, Loss: {loss.item()}')
    # Backward Pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

with torch.no_grad():
    y_eval = model.forward(X_test)
    loss = criterion(y_eval, Y_test)
    
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(X_test):
            y_val = model.forward(data)
                
            print(f'{i+1}.) {str(y_val)}')