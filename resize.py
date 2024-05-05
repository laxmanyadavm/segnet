import os
from PIL import Image

# Path to the directory containing the images and masks
image_dir = 'Mapillary/training/images'
label_dir = 'Mapillary/training/instances'
train_img_output_dir = 'capstone/Dataset/training/images'
train_msk_output_dir = 'capstone/Dataset/training/instances'
val_img_output_dir = 'capstone/Dataset/validation/images'
val_msk_output_dir = 'capstone/Dataset/validation/instances'
test_img_output_dir = 'capstone/Dataset/testing/images'
test_msk_output_dir = 'capstone/Dataset/testing/instances'

resize_dims = (1000, 720)  # Specify the desired dimensions for resizing
max_images = 9000  # Maximum number of image-label pairs to generate

# Create the output directory if it does not exist
os.makedirs(train_img_output_dir, exist_ok=True)
os.makedirs(train_msk_output_dir, exist_ok=True)
os.makedirs(val_img_output_dir, exist_ok=True)
os.makedirs(val_msk_output_dir, exist_ok=True)
os.makedirs(test_img_output_dir, exist_ok=True)
os.makedirs(test_msk_output_dir, exist_ok=True)


# List all files in the image directory
image_files = os.listdir(image_dir)

# Count the number of generated pairs
num_generated = 1

# Loop over each image file
for img_file in image_files:
    if not img_file.endswith('.jpg'):
        continue

    # Check if the pair already exists in the output directory
    label_file = img_file.replace('.jpg', '.png')
    if os.path.exists(os.path.join(train_img_output_dir, img_file)) or os.path.exists(os.path.join(train_msk_output_dir, label_file)):
        print(f"Skipping {img_file} and {label_file} as they already exist in the output folder.")
        continue

    # Open the image and label files
    img_path = os.path.join(image_dir, img_file)
    label_path = os.path.join(label_dir, label_file)

    if not os.path.exists(label_path):
        print(f"Skipping {img_file} because corresponding label {label_file} not found.")
        continue

    img = Image.open(img_path)
    label = Image.open(label_path)

    # Resize the image and label
    resized_img = img.resize(resize_dims, resample=Image.Resampling.NEAREST)
    resized_label = label.resize(resize_dims, resample=Image.Resampling.NEAREST)

    if num_generated<=5000:
    # Save the resized image and label
        resized_img.save(os.path.join(train_img_output_dir, img_file))
        resized_label.save(os.path.join(train_msk_output_dir, label_file))

    elif num_generated>5000 and num_generated<=7000:
        resized_img.save(os.path.join(val_img_output_dir, img_file))
        resized_label.save(os.path.join(val_msk_output_dir, label_file))

    else:
        resized_img.save(os.path.join(test_img_output_dir, img_file))
        resized_label.save(os.path.join(test_msk_output_dir, label_file))

    # Increment the count of generated pairs
    num_generated += 1

    # Break the loop if the maximum number of image-label pairs is reached
    if num_generated > max_images:
        print(f"Generated {num_generated} image-label pairs. Stopping further generation.")
        break
