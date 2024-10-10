import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from model import PRIDNet
from PIL import Image
import os

def calculate_psnr(denoised, target):
  """
  Calculates Peak Signal-to-Noise Ratio (PSNR) between two images.
  Args:
    denoised: Denoised image tensor.
    target: Clean image tensor.
  Returns:
    PSNR value in dB.
  """
  mse = nn.functional.mse_loss(denoised, target)
  max_pixel = 1.0
  psnr = 10 * torch.log10(max_pixel**2 / mse)
  return psnr.item()

def get_data(data_dir):
  """
  Loads and preprocesses image data from a directory.
  Args:
    data_dir: Path to the directory containing images.
  Returns:
    A list of tuples (noisy_image, clean_image).
  """
  data = []
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])
  for filename in os.listdir(data_dir):
    noisy_path = os.path.join(data_dir, filename)
    clean_path = os.path.join(os.path.dirname(data_dir), "train", filename)
    if os.path.isfile(noisy_path) and os.path.isfile(clean_path):
      noisy_image = transform(Image.open(noisy_path).convert('RGB'))
      clean_image = transform(Image.open(clean_path).convert('RGB'))
      data.append((noisy_image, clean_image))
  return data

model = PRIDNet(3, 3)

learning_rate = 0.001
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_data = get_data("data\\train_noisy")
val_data = get_data("data\\train")

train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
val_loader = DataLoader(val_data, batch_size=4)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("loaded data and model")

for epoch in range(epochs):
  model.train()
  train_loss = 0.0
  for noisy_image, clean_image in train_loader:
    noisy_image, clean_image = noisy_image.to(device), clean_image.to(device)

    denoised_image = model(noisy_image)
    loss = criterion(denoised_image, clean_image)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_loss += loss.item()

  model.eval()
  val_loss = 0.0
  psnr = 0.0
  with torch.no_grad():
    for noisy_image, clean_image in val_loader:
      noisy_image, clean_image = noisy_image.to(device), clean_image.to(device)
      denoised_image = model(noisy_image)
      val_loss += criterion(denoised_image, clean_image).item()
      psnr += calculate_psnr(denoised_image, clean_image)

  print(f"Epoch: {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | PSNR: {psnr/len(val_loader):.2f} dB")

model_path = "pridnet_model.pt"
torch.save(model.state_dict(), model_path)
print(f"Model saved to: {model_path}")
