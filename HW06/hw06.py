import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


DEVICE_NUM = 6 
if torch.cuda.is_available():
    device = torch.device(f"cuda:{DEVICE_NUM}")
else:
    device = torch.device("cpu")
print(f"INFO: Using device - {device}")

BATCH_SIZE = 256
PATCH_SIZE = 7
PATCH_INPUT_SIZE = 49 # 7 * 7
SEQUENCE_LENGTH = 16 # 28*28 / 7*7
HIDDEN_SIZE = 128
NUM_CLASSES = 10
NUM_EPOCHES = 15
LEARNING_RATE = 3e-4


def image_to_patches(image_tensor):
    patches = image_tensor.unfold(1, PATCH_SIZE, PATCH_SIZE) # (1, 4, 28, 7)
    patches = patches.unfold(2, PATCH_SIZE, PATCH_SIZE) # (1, 4, 4, 7, 7)
    patches = patches.squeeze().contiguous() # (4, 4, 7, 7)
    patches = patches.view(SEQUENCE_LENGTH, PATCH_INPUT_SIZE) # (16, 49)
    return patches


transform_with_patches = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(image_to_patches)
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform_with_patches)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform_with_patches)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)


class MNISTRnn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        last_out = out[:, -1, :]
        out = self.linear(last_out)
        return out


model = MNISTRnn(input_size=PATCH_INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=NUM_CLASSES)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("INFO: Start Training")
for epoch in range(NUM_EPOCHES): 
    model.train()
    running_loss = 0.0
    for sequences, labels in train_loader: 
        sequences = sequences.to(device) 
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
print("INFO: Training Complete")

model.eval()
with torch.no_grad():
    total_prediction = 0
    correct_prediction = 0
    for sequences, labels in test_loader:
        sequences = sequences.to(device) 
        labels = labels.to(device)
        outputs = model(sequences)
        _, predicted = torch.max(outputs, dim=1)
        total_prediction += labels.size(0)
        correct_prediction += (predicted == labels).sum().item()
print(f"Total: {total_prediction}, Correct: {correct_prediction}, Accuracy: {correct_prediction/total_prediction:.2%}")
