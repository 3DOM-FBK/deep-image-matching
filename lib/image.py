from pathlib import Path

class Image:
    def __init__(self, img_id : int, absolute_path : Path):
        self.id = img_id
        self.name = absolute_path.stem
        self.absolute_path = absolute_path