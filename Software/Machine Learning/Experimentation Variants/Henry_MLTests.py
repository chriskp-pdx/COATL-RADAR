# ML Program, Chris Kane-Pardy

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Load Data File (File with Training Data)
TrainingData = r"C:\Users\henry\OneDrive\Desktop\PSU\Winter 24-25\ECE 412_413 - Senior Project Development\COATL-RADAR\pyFiles\Datasets\MedBeanis_Testing V2 - BEANSV.csv"
df = pd.read_csv(TrainingData)

# 2. Map Bean Names → Percent Value for Clarity
Index = {'Yemen': 2, 'Rwanda': 1, 'Guatemala': 0}
InverseIndex = {v: k for k, v in Index.items()}
df['Bean Name'] = df['Bean Name'].map(Index)

# 3. Feature / target split
df = df.drop(columns=["Scan_Group"])
BeanValue = df[[f"Point_{i}" for i in range(50)]].values.astype(np.float32)
BeanName = df['Bean Name'].values.astype(np.int64)     # shape (N,), Moisture %

# 4. (Optional) Label encoding for convenience
#label_encoder = LabelEncoder()
#y_encoded = label_encoder.fit_transform(y)

# 5. Train / test split (80% train, 20% test)
BeanValueTrain, BeanValueTest, BeanNameTrain, BeanNameTest = train_test_split(
    BeanValue, BeanName, test_size=0.2, random_state=12 )
# The split is between data used to train the model, and data used for the model to test its accuracy
# The random seed only determines what values will be used for each

# 6. Convert to PyTorch tensors
BeanValueTrain = torch.from_numpy(BeanValueTrain)
BeanValueTest  = torch.from_numpy(BeanValueTest)
BeanNameTrain  = torch.from_numpy(BeanNameTrain)
BeanNameTest   = torch.from_numpy(BeanNameTest)

# 7. Define the feed‑forward classifier
class Model(nn.Module):
    def __init__(self, in_features=50, h1=512, h2=256, h3=128, h4=64, h5=32, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2,h3)
        self.fc4 = nn.Linear(h3,h4)
        self.fc5 = nn.Linear(h4,h5)
        self.out = nn.Linear(h5, out_features)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return self.out(x)

# 8. Instantiate model, loss, optimizer
torch.manual_seed(30)
model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 9. Training loop
epochs = 2000
losses = []
for epoch in range(epochs):
    logits = model(BeanValueTrain)
    loss = criterion(logits, BeanNameTrain)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 500 == 0:
        print(f"Epoch {epoch:5d}  Loss: {loss.item():.4f}")

# 10. Plot training loss curve
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

# 11. Evaluate on test set
with torch.no_grad():
    test_logits = model(BeanValueTest)
    test_loss = criterion(test_logits, BeanNameTest).item()
    preds = torch.argmax(test_logits, dim=1)
    accuracy = (preds == BeanNameTest).float().mean().item() * 100
print(f"Test loss: {test_loss:.4f}   Accuracy: {accuracy:.1f}%\n")

# 12. Prediction helper
def predict_bean(bean_value: float) -> str:
    model.eval()
    x = torch.tensor([[bean_value]], dtype=torch.float32)
    with torch.no_grad():
        logits = model(x)
    idx = torch.argmax(logits, dim=1).item()
    return InverseIndex[idx]

# 13. Interactive REPL for predictions
if __name__ == "__main__":
    print("Enter a bean value to predict its origin, or 'q' to quit.")
    while True:
        s = input("Bean Value ▶ ").strip()
        if s.lower() in ('q', 'quit', 'exit'):
            print("Exiting. Goodbye!")
            break
        try:
            val = float(s)
        except ValueError:
            print("  ✗ Invalid number, try again.")
            continue
        pred_name = predict_bean(val)
        print(f"  ✓ Predicted Bean Name: {pred_name}\n")

