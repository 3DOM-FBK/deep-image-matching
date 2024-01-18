import importlib

# Try importing deep_image_matching package.


def main():
    try:
        importlib.import_module("deep_image_matching")
    except ImportError:
        raise ImportError("Unable to import ICEpy4D module")


if __name__ == "__main__":
    main()
