import os
import shutil


def save_every_x_image(source_folder, output_folder, x):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all files in the source folder
    files = os.listdir(source_folder)

    # Iterate through the files and copy every x-th image to the output folder
    for i, file_name in enumerate(files):
        if i % x == 0:
            source_path = os.path.join(source_folder, file_name)
            output_path = os.path.join(output_folder, file_name)
            shutil.copy2(source_path, output_path)
            print(f"Copying {file_name} to {output_folder}")


# Replace 'source_folder', 'output_folder', and 'x' with your actual values
source_folder = "./full_res/images"
output_folder = "./skip"
x = 10  # Change this to the interval you desire

save_every_x_image(source_folder, output_folder, x)
