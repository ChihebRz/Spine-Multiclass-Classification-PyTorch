import torch
import torch.nn as nn
from preprocess import load_data
from model import SpineMulticlassClassifier


# ----- Load data -----
X_train, X_test, y_train, y_test = load_data("../data/column_3C_weka.csv")

# ----- Model -----
model = SpineMulticlassClassifier()

# ----- Loss & optimizer -----
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 2000

# ----- Training loop -----
for epoch in range(epochs):
    model.train()

    logits = model(X_train)
    loss = criterion(logits, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.4f}")

# ----- Save -----
torch.save(model.state_dict(), "../artifacts/model.pth")
print("Model saved to artifacts/model.pth")
