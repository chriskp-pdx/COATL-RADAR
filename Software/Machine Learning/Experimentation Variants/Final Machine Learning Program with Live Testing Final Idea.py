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

# 1. Load Data File
TrainingData = r"C:\Users\Chris\Desktop\Git4School\COATL-RADAR\Software\Data Collection\Datasets\CombinedBeanDataAll3 - Test.csv"
df = pd.read_csv(TrainingData)

# 2. Map Bean Names → Percent Value
Index = {'Yemen': 2, 'Rwanda': 1, 'Guatemala': 0}
InverseIndex = {v: k for k, v in Index.items()}
df['Bean Name'] = df['Bean Name'].map(Index)

# 3. Feature / target split
df = df.drop(columns=["Scan_Group"])
df = df.drop(columns=[f"Point_{i}" for i in range(100)])
BeanName = df['Bean Name'].values.astype(np.int64)
BeanValue = df['Overall Mean'].values.astype(np.float32)

# 4. Train / test split
BeanValueTrain, BeanValueTest, BeanNameTrain, BeanNameTest = train_test_split(
    BeanValue, BeanName, test_size=0.2, random_state=15
)

# 5. Convert to PyTorch tensors
BeanValueTrain = torch.from_numpy(BeanValueTrain).unsqueeze(1)
BeanValueTest = torch.from_numpy(BeanValueTest).unsqueeze(1)
BeanNameTrain = torch.from_numpy(BeanNameTrain)
BeanNameTest = torch.from_numpy(BeanNameTest)

# 6. Define the model
class Model(nn.Module):
    def __init__(self, in_features=1, h1=256, h2=128, h3=64, h4=32, h5=16, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, h4)
        self.fc5 = nn.Linear(h4, h5)
        self.out = nn.Linear(h5, out_features)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return self.out(x)

# 7. Train the model
torch.manual_seed(30)
model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 3000
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

# 8. Plot training loss
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

# 9. Evaluate on test set
with torch.no_grad():
    test_logits = model(BeanValueTest)
    test_loss = criterion(test_logits, BeanNameTest).item()
    preds = torch.argmax(test_logits, dim=1)
    accuracy = (preds == BeanNameTest).float().mean().item() * 100
print(f"Test loss: {test_loss:.4f}   Accuracy: {accuracy:.1f}%\n")

# 10. Sensor scan functions
def MultiScanAverage(client, sensor_config, Scans, Delay=0.0005):
    total_amplitudes = np.zeros(sensor_config.num_points, dtype=np.complex128)
    client.setup_session(sensor_config)
    client.start_session()
    for _ in range(Scans):
        data = client.get_next()
        amplitudes = np.array(data.frame[0].tolist())
        total_amplitudes += amplitudes
        time.sleep(Delay)
    client.stop_session()
    return total_amplitudes / Scans

def run_single_group_scan(serial_port="COM5"):
    client = a121.Client.open(serial_port=serial_port)

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

    print("Starting calibration scan...")
    CalibrationAmplitudes = MultiScanAverage(client, sensor_config, Scans=50)
    print("Calibration scan complete.")

    input("Insert beans and press Enter to continue...")

    print("Starting bean scan group...")
    BeanData = MultiScanAverage(client, sensor_config, Scans=50)
    print("Bean scan complete.")

    client.close()

    diff = CalibrationAmplitudes - BeanData
    real_part = np.real(diff)
    imag_part = np.imag(diff)
    real_avg = np.average(real_part)
    imag_avg = np.average(imag_part)
    mean_value = np.average(real_avg,imag_avg)
    return mean_value

# 11. Prediction helper
def predict_bean(bean_value: float) -> str:
    model.eval()
    x = torch.tensor([[bean_value]], dtype=torch.float32)
    with torch.no_grad():
        logits = model(x)
    idx = torch.argmax(logits, dim=1).item()
    return InverseIndex[idx]

# 12. Interactive REPL
if __name__ == "__main__":
    print("Bean Scanner Ready! Press Enter to scan, or type 'q' to quit.\n")
    while True:
        s = input("▶ Please Empty Your Device for a Calibration Scan, and Press Enter").strip()
        if s.lower() in ('q', 'quit', 'exit'):
            print("Exiting. Goodbye!")
            break
        print("  → Starting scan...")
        bean_value = run_single_group_scan()
        pred_name = predict_bean(bean_value)
        print(f"  ✓ Predicted Bean Name: {pred_name}\n")
