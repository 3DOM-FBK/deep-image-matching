import cv2
import os
import argparse

def extract_frames(video_path, output_folder, interval):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        exit()

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 0
    while True:
        ret, frame = cap.read()

        # Check if the video has ended
        if not ret:
            break

        # Save frames at the specified interval
        if frame_count % interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)

        frame_count += 1

    # Release the video capture object
    cap.release()

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Extract frames from a video at a specified time interval.")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("output_folder", help="Path to the output folder for extracted frames")
    parser.add_argument("--interval", type=int, default=25, help="Time interval between frames (default: 25 frames)")

    args = parser.parse_args()

    # Extract frames
    extract_frames(args.video_path, args.output_folder, args.interval)

    print(f"Frames extracted successfully to {args.output_folder}")

if __name__ == "__main__":
    main()
