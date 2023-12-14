# %%
from pathlib import Path

data_path = Path("data/")
image_path = data_path / "events"
train_dir = image_path / "train"
test_dir = image_path / "test"

train_dir, test_dir
output_dir = 'output'

# %%
import random
from PIL import Image
# Set seed
random.seed(42) # <- try changing this and see what happens

# %%

import torch
from torchvision import transforms
torch.manual_seed(42)

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
device

# %%
import matplotlib.pyplot as plt
import torch
import torchvision

from torch import nn
from torchvision import transforms

from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s
weights = EfficientNet_V2_S_Weights.DEFAULT
model = efficientnet_v2_s(weights=weights)


# %%
data_transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=False), 
    transforms.ToTensor(), # 2. Turn image values to between 0 & 1 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # 3. A mean of [0.485, 0.456, 0.406] (across each colour channel)
                         std=[0.229, 0.224, 0.225]) # 4. A standard deviation of [0.229, 0.224, 0.225] (across each colour channel),
])
auto_transforms = weights.transforms()

# %%
data_transform2 = transforms.Compose([
    transforms.RandomResizedCrop(224),  # Randomly crop and resize the image
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust color
    transforms.RandomRotation(30),  # Randomly rotate the image up to 30 degrees
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Randomly translate the image
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image data
])

# %%
from torchvision import datasets
train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                  transform=data_transform, # transforms to perform on data (images)
                                  target_transform=None) # transforms to perform on labels (if necessary)
test_data = datasets.ImageFolder(root=test_dir, 
                                 transform=data_transform)
train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                  transform=auto_transforms, # transforms to perform on data (images)
                                  target_transform=None) # transforms to perform on labels (if necessary)
test_data = datasets.ImageFolder(root=test_dir, 
                                 transform=auto_transforms)


# %%

class_names = train_data.classes
class_names

# %%
from torch.utils.data import DataLoader
train_dataloader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=32,shuffle=False)

# %%
# # Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
for param in model.features.parameters():
    param.requires_grad = False

# %%
# Get the length of class_names (one output unit for each class)
output_shape = len(class_names)
# torch.autograd.set_detect_anomaly(True)
# Recreate the classifier layer and seed it to the target device

model.classifier = torch.nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=output_shape, bias=True),
    # nn.ReLU6(),   
).to(device)

# %%
from torch import nn
loss_fn = nn.CrossEntropyLoss() # this is also called "criterion"/"cost function" in some places
optimizer = torch.optim.RMSprop(params=model.parameters(), lr=0.003)

# %%
# Set the random seeds
try:
    from going_modular.going_modular import engine
except:
    # Get the going_modular scripts
    print("[INFO] Couldn't find going_modular scripts... downloading them from GitHub.")
    !git clone https://github.com/mrdbourke/pytorch-deep-learning
    !move pytorch-deep-learning\going_modular .
    !rmdir /s /q pytorch-deep-learning
    from going_modular.going_modular import engine

# Start the timer
from timeit import default_timer as timer 
start_time = timer()

# Setup training and save the results
results = engine.train(model=model,
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       epochs=3,
                       device=device)

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

# %%
from typing import List, Tuple

from PIL import Image



# %%
torch.save(model, 'model.tf')
torch.save(model.state_dict(), 'weights.tf')
