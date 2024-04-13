"""Database for sensor-width.

This database if used when we construct camera intrinsics from exif data.

Authors: Ayush Baid
"""

import csv
import shutil
import urllib.request
from pathlib import Path


class SensorWidthDatabase:
    """Database class for sensor-width, reading data from a csv file."""

    DEFAULT_SENSOR_DB_PATH = Path(__file__).resolve().parents[1] / "thirdparty/sensor_width_camera_database.csv"
    DATABASE_URL = url = (
        "https://raw.githubusercontent.com/openMVG/openMVG/6d6b1dd70bded094ba06024e481dd5a5c662dc83/src/openMVG/exif/sensor_width_database/sensor_width_camera_database.txt"
    )
    DATABASE_DETAIL_ULR = ""

    def __init__(self, csv_path: str = None):
        """Initializes the database from a csv file"""

        if csv_path is None or not Path(csv_path).exists():
            csv_path = Path(self.DEFAULT_SENSOR_DB_PATH)
            url = self.DATABASE_URL
            with urllib.request.urlopen(url) as response, open(csv_path, "wb") as out_file:
                shutil.copyfileobj(response, out_file)

        if not csv_path.exists():
            raise FileNotFoundError(f"Sensor-width database not found at '{csv_path}'")

        # Store data in a dictionary for efficient lookup
        self.data = {}
        with open(csv_path, "r") as file:
            reader = csv.reader(file, delimiter=";")
            for row in reader:
                try:
                    key = row[0].lower()
                    self.data[key] = float(row[1])
                except ValueError:
                    continue

    def lookup(self, camera: str) -> float:
        """Look-up the sensor width given the camera make and model.

        Args:
            camera: camera model

        Returns:
            sensor-width in mm
        """

        # preprocess query strings
        key = camera.lower()
        if key not in self.data:
            raise LookupError(f"Camera {key} not found in sensor database")

        return self.data[key]


if __name__ == "__main__":
    db = SensorWidthDatabase()

    db.lookup("Canon EOS 5D Mark II")

    print("done.")
