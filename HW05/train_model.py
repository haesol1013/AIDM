import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from model import MLP


BATCH_SIZE = 32
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = MLP()

NUM_EPOCHS = 10
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("INFO: Start Training")
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0

    for data in train_loader:
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

print("INFO: Training Complete")

out_path = "model.pt"
torch.save(model.state_dict(), f=out_path)
