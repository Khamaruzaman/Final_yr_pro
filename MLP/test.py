import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from mlp import FusionMLP

# 🔹 Load the trained model
model = FusionMLP()
model.load_state_dict(torch.load("fusion_mlp.pth"))
model.eval()  # Set model to evaluation mode

# 🔹 Function to get user input and make predictions
def test_model():
    while True:
        try:
            s_branch = float(input("Enter S-Branch prediction (0-1): "))
            l_branch = float(input("Enter L-Branch prediction (0-1): "))
            
            if not (0 <= s_branch <= 1 and 0 <= l_branch <= 1):
                print("Invalid input! Please enter values between 0 and 1.")
                continue

            # Convert input to tensor
            input_tensor = torch.tensor([[s_branch, l_branch]], dtype=torch.float32)

            # Get model prediction
            with torch.no_grad():
                prediction = model(input_tensor).item()
            
            print(f"Model Output: {prediction:.4f}")
            print(f"Predicted Label: {'Real (1)' if prediction < 0.5 else 'Fake (0)'}\n")
        
        except ValueError:
            print("Invalid input! Please enter numerical values between 0 and 1.")
        
        # Ask if user wants to continue
        cont = input("Test another input? (y/n): ").strip().lower()
        if cont != 'y':
            print("Exiting...")
            break

# Run the testing function
test_model()
