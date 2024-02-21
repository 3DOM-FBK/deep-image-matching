import cv2
import os

# Function to create video from images in a folder
def images_to_video(image_folder, output_video_path, frame_duration=0.5):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))

    # Get image dimensions
    height, width, layers = frame.shape

    # Define the video codec and create VideoWriter object
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
    #fourcc = cv2.CV_FOURCC(-1)  # You can try 'H264' as well
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    video = cv2.VideoWriter(output_video_path, fourcc, 1 / frame_duration, (width, height))

    for i, image in enumerate(images):
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)

        # Resize the image to match the dimensions of the first image (if needed)
        # frame = cv2.resize(frame, (width, height))

        # Write the frame to the video file
        video.write(frame)
        print(f"Processed frame {i}")

    # Release the VideoWriter object
    video.release()

# Example usage
images_folder = r'C:\Users\threedom\Desktop\DIM_VIDEO\semperoper\superglue\sift_screen_resized'
output_video_path = r'C:\Users\threedom\Desktop\DIM_VIDEO\semperoper\superglue\sift_screen_resized\semperoper_sift.avi'
frame_duration = 0.5  # seconds

images_to_video(images_folder, output_video_path, frame_duration)