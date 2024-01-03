import os
import cv2
import numpy as np
import albumentations as A
import RRDBNet_arch as arch
import torch

def decrease_resolution(image, target_width, target_height):
    dim = (target_width, target_height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

def process_images(input_folder, output_folder, target_width, target_height):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
    model_path = 'RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth

    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    # Process each subfolder in the input folder
    for subfolder in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder)
        if os.path.isdir(subfolder_path):
            output_subfolder = os.path.join(output_folder, subfolder)
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)

            # Process each image in the subfolder
            for filename in os.listdir(subfolder_path):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    input_path = os.path.join(subfolder_path, filename)
                    output_path = os.path.join(output_subfolder, filename) 
                    original = cv2.imread(input_path)
                    scaled_image = decrease_resolution(original, target_width, target_height)
                    cv2.imwrite(output_path, scaled_image)

                    img = cv2.imread(output_path, cv2.IMREAD_COLOR)
                    img = img * 1.0 / 255
                    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
                    img_LR = img.unsqueeze(0)
                    img_LR = img_LR.to(device)                                 
                    with torch.no_grad():
                        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
                    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
                    output = (output * 255.0).round()
                    cv2.imwrite(output_path, output)

train_input_folder = r"data\events\train"
train_output_folder = r"data\events\train_scaled"

test_input_folder = r'data\events\test'
test_output_folder = r'data\events\test_scaled'

# Set scale and noise parameters
target_width = 100
target_height = 100
noise_level = 200

# Process train images
process_images(train_input_folder, train_output_folder, target_width, target_height)

# Process test images
process_images(test_input_folder, test_output_folder, target_width, target_height)












