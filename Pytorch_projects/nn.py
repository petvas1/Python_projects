# Import necessary libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

# Load the dataset from CSV
df = pd.read_csv("data.csv").dropna()  # Remove missing values

X = df.drop(columns=['t', 'time'], axis=1).values
y = df['t'].values


# Standardize the features (important for neural networks)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data: 70% training, 15% validation, 15% testing
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert data into PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_valid = torch.tensor(X_valid, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # Reshape for single output
y_valid = torch.tensor(y_valid, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


# Define a deeper neural network with improved hyperparameters
class TemperatureNN(nn.Module):
    def __init__(self, input_dim):
        super(TemperatureNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)  # Increased neurons
        self.fc2 = nn.Linear(256, 256)  # Second hidden layer (256 neurons)
        self.fc3 = nn.Linear(256, 128)  # Third hidden layer (128 neurons)
        self.fc4 = nn.Linear(128, 1)  # Output layer (single neuron for regression)
        self.leaky_relu = nn.LeakyReLU()  # Better than ReLU for small gradients
        self.dropout = nn.Dropout(0.3)  # Dropout to reduce overfitting

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)  # Dropout applied after activation
        x = self.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc3(x))
        x = self.fc4(x)  # No activation in the output layer for regression
        return x


# Initialize model
input_dim = X_train.shape[1]  # Number of input features
model = TemperatureNN(input_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss for regression
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Lower learning rate

# Training parameters
num_epochs = 1000  # More epochs for better learning
batch_size = 32  # Mini-batch training

# Create DataLoaders for mini-batch training
train_data = torch.utils.data.TensorDataset(X_train, y_train)
valid_data = torch.utils.data.TensorDataset(X_valid, y_valid)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)

# Training loop with validation set
train_losses, valid_losses = [], []

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    epoch_train_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()  # Reset gradients
        y_pred = model(batch_X)  # Forward pass
        loss = criterion(y_pred, batch_y)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        epoch_train_loss += loss.item()

    # Compute validation loss
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        valid_loss = sum(criterion(model(X_valid), y_valid).item() for _ in valid_loader) / len(valid_loader)

    # Store losses
    train_losses.append(epoch_train_loss / len(train_loader))
    valid_losses.append(valid_loss)

    # Print progress every 50 epochs
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Valid Loss: {valid_losses[-1]:.4f}")

# Evaluate model on test data
model.eval()
y_pred_nn = model(X_test).detach().numpy()

# Compute performance metrics
mae = mean_absolute_error(y_test, y_pred_nn)
mse = mean_squared_error(y_test, y_pred_nn)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_nn)
explained_var = explained_variance_score(y_test, y_pred_nn)

# Print results
print("\n🔹 Neural Network Performance:")
print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R² Score: {r2:.3f}")
print(f"Explained Variance Score: {explained_var:.3f}")

# Plot training and validation loss
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Training Loss", color="blue")
plt.plot(valid_losses, label="Validation Loss", color="red")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title("Training & Validation Loss Over Epochs")
plt.legend()
plt.grid()
plt.show()
