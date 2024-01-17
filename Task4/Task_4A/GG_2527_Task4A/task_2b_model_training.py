# %%
from pathlib import Path

data_path = Path("data/")
image_path = data_path / "events"
train_dir = image_path / "train_scaled"
test_dir = image_path / "test_scaled"

train_dir, test_dir
output_dir = 'output'

# %%

import torch
torch.cuda.empty_cache()
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# %%
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

# %%


# %%
import matplotlib.pyplot as plt
import torch

from torch import nn
from torchvision import transforms

from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

weights = EfficientNet_V2_S_Weights.DEFAULT
model = efficientnet_v2_s(weights=weights).to(device)

# %%
auto_transforms = weights.transforms()

# %%
from torchvision import datasets
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
train_dataloader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=16,shuffle=False)

# %%
# Get the length of class_names (one output unit for each class)
output_shape = len(class_names)
# Recreate the classifier layer and seed it to the target device

model.classifier = torch.nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=output_shape, bias=True),
).to(device)

# %%
from torch import nn
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1) # this is also called "criterion"/"cost function" in some places
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

# %%
torch.cuda.set_device(0)

# %%
next(model.parameters()).is_cuda

# %%
for layer in model.features.parameters():
    layer.requires_grad = True

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
                       epochs=15,
                       device=device)

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")


# %%
torch.save(model.state_dict(), 'w3.tf')


