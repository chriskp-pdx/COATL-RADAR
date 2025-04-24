import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Load your uploaded data
DATA_PATH = "C:\Acconeer Python\COATL-RADAR\pyFiles\Dataset\BeanMaxNormalized_Output_ConstantAmplitudeV4 - ToExport4.csv"
df = pd.read_csv(DATA_PATH)

# 2. Map bean names → integers
name_map = {'Yemen': 0, 'Rwanda': 1, 'Guatemala': 2}
inv_name_map = {v: k for k, v in name_map.items()}
df['Bean Name'] = df['Bean Name'].map(name_map)

# 3. Feature / target split
X = df[['Bean Value']].values.astype(np.float32)  # shape (N,1)
y = df['Bean Name'].values.astype(np.int64)       # shape (N,)

# 4. (Optional) Label encoding for convenience
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 5. Train / test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=30
)

# 6. Convert to PyTorch tensors
X_train = torch.from_numpy(X_train)
X_test  = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train)
y_test  = torch.from_numpy(y_test)

# 7. Define the feed‑forward classifier
class Model(nn.Module):
    def __init__(self, in_features=1, h1=64, h2=32, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

# 8. Instantiate model, loss, optimizer
torch.manual_seed(30)
model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# 9. Training loop
epochs = 20000
losses = []
for epoch in range(epochs):
    logits = model(X_train)
    loss = criterion(logits, y_train)
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
    test_logits = model(X_test)
    test_loss = criterion(test_logits, y_test).item()
    preds = torch.argmax(test_logits, dim=1)
    accuracy = (preds == y_test).float().mean().item() * 100
print(f"Test loss: {test_loss:.4f}   Accuracy: {accuracy:.1f}%\n")

# 12. Prediction helper
def predict_bean(bean_value: float) -> str:
    model.eval()
    x = torch.tensor([[bean_value]], dtype=torch.float32)
    with torch.no_grad():
        logits = model(x)
    idx = torch.argmax(logits, dim=1).item()
    return inv_name_map[idx]

# 13. Interactive REPL for predictions
if __name__ == "__main__":
    print("=== Bean Classifier REPL ===")
    print("Enter a bean value (300–8000) to predict its origin, or 'q' to quit.")
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
        if not (300 <= val <= 8000):
            print("  ⚠️  Value outside 300–8000 range. Predicting anyway...")
        pred_name = predict_bean(val)
        print(f"  ✓ Predicted Bean Name: {pred_name}\n")

