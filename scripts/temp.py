import cv2
import matplotlib.pyplot as plt


def plot_intensity_curve(image_path):
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Error: Could not read the image.")
        return

    # img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    mean_intensity = img.mean()
    std_intensity = img.std()

    # Z-score normalization
    # img = ((img - mean_intensity) / std_intensity).astype('uint8')
    cv2.imshow("Original Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Calculate the intensity histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    # Plot the intensity curve
    plt.plot(hist)
    plt.title("Intensity Curve")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()


if __name__ == "__main__":
    # Specify the path to your image
    image_path = (
        r"C:\Users\threedom\Desktop\Neil\luca_all\images\11-12-22-313-radiometric.jpg"
    )

    # Call the function to plot the intensity curve
    plot_intensity_curve(image_path)
