import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from model import MLP


BATCH_SIZE = 32
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

out_path = "model.pt"
model = MLP()
model.load_state_dict(torch.load(out_path))

with torch.no_grad():
    total_prediction = 0
    correct_prediction = 0

    for data in test_loader:
        inputs, labels = data
        outputs = model(inputs)
        _, predicted = torch.max(outputs, dim=1)
        total_prediction += labels.size(0)
        correct_prediction += (predicted == labels).sum().item()

print(f"Total: {total_prediction}, Correct: {correct_prediction}, Accuracy: {correct_prediction/total_prediction:.2%}")
