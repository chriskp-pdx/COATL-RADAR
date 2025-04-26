# ML Bean Moisture Detection Program, Chris Kane-Pardy

import acconeer.exptool as et
from acconeer.exptool import a121
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Load Data File (File with Training Data)
TrainingData = r"C:\Users\HP\OneDrive\Desktop\Git 4 School\COATL-RADAR\pyFiles\Datasets\MedBeanis_Testing V2 - BEANSV2.csv"
df = pd.read_csv(TrainingData)

# 2. Map Bean Names → Percent Value for Clarity
Index = {'Yemen': 2, 'Rwanda': 1, 'Guatemala': 0}
InverseIndex = {v: k for k, v in Index.items()}
df['Bean Name'] = df['Bean Name'].map(Index)

# 3. Feature / target split
df = df.drop(columns=["Scan_Group"])
BeanValue = df[[f"Point_{i}" for i in range(100)]].values.astype(np.float32)
BeanName = df['Bean Name'].values.astype(np.int64)     # shape (N,), Moisture %

# 5. Train / test split (80% train, 20% test)
BeanValueTrain, BeanValueTest, BeanNameTrain, BeanNameTest = train_test_split(
    BeanValue, BeanName, test_size=0.2, random_state=15 )
# The split is between data used to train the model, and data used for the model to test its accuracy
# The random seed only determines what values will be used for each

# 6. Convert to PyTorch tensors
BeanValueTrain = torch.from_numpy(BeanValueTrain)
BeanValueTest  = torch.from_numpy(BeanValueTest)
BeanNameTrain  = torch.from_numpy(BeanNameTrain)
BeanNameTest   = torch.from_numpy(BeanNameTest)

# 7. Define the feed‑forward classifier
class Model(nn.Module):
    def __init__(self, in_features=100, h1=512, h2=256, h3=128, h4=64, h5=32, out_features=3):
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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-8)

# 9. Training loop
epochs = 40000
losses = []
for epoch in range(epochs):
    logits = model(BeanValueTrain)
    loss = criterion(logits, BeanNameTrain)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if epoch % 100 == 0:
        print(f"Epoch {epoch:5d}  Loss: {loss.item():.8f}")

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

# 12. Bean Scanning Program
def MultiScanAverage(client, sensor_config, Scans, Delay=0.0005):
    total_amplitudes = np.zeros(sensor_config.num_points, dtype=np.complex128)

    for _ in range(Scans):
        client.start_session()
        data = client.get_next()
        client.stop_session()

        amplitudes = np.array(data.frame[0].tolist())
        total_amplitudes += amplitudes
        
        time.sleep(Delay)
    
    return total_amplitudes / Scans

def run_single_group_scan(serial_port="COM5"):
    # Setup sensor client
    client = a121.Client.open(serial_port=serial_port)

    # Define sensor configuration
    sensor_config = a121.SensorConfig()
    sensor_config.step_length = 1
    sensor_config.start_point = 40
    sensor_config.num_points = 50
    sensor_config.sweeps_per_frame = 1
    sensor_config.hwaas = 500
    sensor_config.profile = et.a121.Profile.PROFILE_1
    sensor_config.prf = 19.5e6
    sensor_config.receiver_gain = 19
    sensor_config.phase_enhancement = True

    # Calibration scan
    client.setup_session(sensor_config)
    print("Starting calibration scan...")
    CalibrationAmplitudes = MultiScanAverage(client, sensor_config, Scans=50)
    print("Calibration scan complete.")

    input("Insert beans and press Enter to continue...")

    # Bean scan (1 group of 50 scans)
    print("Starting bean scan group...")
    client.setup_session(sensor_config)
    BeanData = MultiScanAverage(client, sensor_config, Scans=50)
    print("Bean scan complete.")

    client.close()

    # Real + Imag → 1x100 vector
    diff = CalibrationAmplitudes - BeanData
    real_part = np.real(diff)
    imag_part = np.imag(diff)
    combined = np.zeros
    combined = np.hstack((real_part, imag_part)).astype(np.float32)
    return combined

# 12. Prediction helper
def predict_bean(bean_value: np.ndarray) -> str:
    model.eval()                              # sets model to inference (not training) mode
    x = torch.tensor(bean_value, dtype=torch.float32).unsqueeze(0)  # add batch dim (1, 100)
    with torch.no_grad():                     # disables gradient calc for faster inference
        logits = model(x)                     # forward pass through the model
    idx = torch.argmax(logits, dim=1).item()  # index of highest score → class prediction
    return InverseIndex[idx]                  # convert index back to readable bean name

# 13. Interactive REPL for predictions
if __name__ == "__main__":
    print("Bean Scanner Ready! Press Enter to scan, or type 'q' to quit.\n")
    
    while True:
        s = input("▶ Please Empty Your Device for a Calibration Scan, and Press Enter").strip()
        if s.lower() in ('q', 'quit', 'exit'):
            print("Exiting. Goodbye!")
            break

        print("  → Starting scan...")
        bean_vector = run_single_group_scan()
        pred_name = predict_bean(bean_vector)
        print(f"  ✓ Predicted Bean Name: {pred_name}\n")


