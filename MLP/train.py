import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from mlp import FusionMLP

# 🔹 1️⃣ Load CSV Data
csv_file = "predictions.csv"  # Change this to your actual CSV file path
df = pd.read_csv(csv_file)

# 🔹 2️⃣ Extract Features & Labels
s_branch_preds = torch.tensor(df.iloc[:, 1].values, dtype=torch.float32).unsqueeze(1)  # S-Branch
l_branch_preds = torch.tensor(df.iloc[:, 2].values, dtype=torch.float32).unsqueeze(1)  # L-Branch
labels = torch.tensor(df.iloc[:, 3].values, dtype=torch.float32).unsqueeze(1)  # Ground Truth (Real/Fake)

# 🔹 3️⃣ Train-Test Split
X = torch.cat((s_branch_preds, l_branch_preds), dim=1)  # Combine features
y = labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch Dataset
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# 🔹 4️⃣ Instantiate Model

model = FusionMLP()
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 🔹 5️⃣ Training Loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Print every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

# 🔹 6️⃣ Evaluate on Test Data
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        predictions = model(batch_X)
        predicted_labels = (predictions > 0.5).float()  # Threshold 0.5
        correct += (predicted_labels == batch_y).sum().item()
        total += batch_y.size(0)

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# 🔹 7️⃣ Save the Model
torch.save(model.state_dict(), "fusion_mlp.pth")
print("Model saved as fusion_mlp.pth")
