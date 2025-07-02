import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def plot_intensity_curve(image_path):
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Error: Could not read the image.")
        return

    # Calculate mean and standard deviation of pixel intensities
    mean_intensity = img.mean()
    std_intensity = img.std()

    # Z-score normalization
    # img_normalized = ((img - mean_intensity) / std_intensity).astype('uint8')
    img_normalized = img

    # Calculate the intensity histogram for normalized image
    hist, bin_edges = np.histogram(img_normalized.flatten(), bins=256, range=[0, 256])

    # Find the bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Define the Gaussian fit function
    def gaussian(x, amplitude, mean, stddev):
        return amplitude * np.exp(-(((x - mean) / 4 / stddev) ** 2))

    # Fit the Gaussian curve to the histogram data
    initial_guess = [hist.max(), bin_centers[np.argmax(hist)], 20]
    popt, _ = curve_fit(gaussian, bin_centers, hist, p0=initial_guess)

    # Filter out outliers (e.g., keep values within 3 standard deviations from the mean)
    outlier_threshold = 3 * popt[2]  # 3 times the standard deviation
    filtered_data = img_normalized[np.abs(img_normalized - popt[1]) < outlier_threshold]

    # Plot the intensity histogram
    plt.plot(bin_centers, hist, label="Intensity Histogram")

    # Plot the Gaussian fit
    plt.plot(bin_centers, gaussian(bin_centers, *popt), "r-", label="Gaussian Fit")

    # Plot the filtered data (outliers removed)
    plt.hist(
        filtered_data.flatten(),
        bins=256,
        range=[0, 256],
        alpha=0.5,
        color="green",
        label="Filtered Data",
    )

    plt.title("Intensity Curve with Gaussian Fit")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Specify the path to your image
    image_path = (
        r"C:\Users\threedom\Desktop\Neil\luca_all\images\11-12-22-313-radiometric.jpg"
    )

    # Call the function to plot the intensity curve with Gaussian fit and outlier rejection
    plot_intensity_curve(image_path)
