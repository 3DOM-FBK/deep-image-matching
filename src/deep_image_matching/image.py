from pathlib import Path


class Image:
    def __init__(self, img_id: int, absolute_path: Path):
        self.id = img_id
        self.name = absolute_path.stem
        self.absolute_path = absolute_path


class ImageList:
    def __init__(self, img_dir: Path):
        self.images = []
        self.current_idx = 0
        i = 0
        all_imgs = [
            image
            for image in img_dir.glob("*")
            if image.suffix in [".jpg", ".JPG", ".png"]
        ]
        all_imgs.sort()

        for image in all_imgs:
            self.add_image(i, image)
            i += 1

    def __len__(self):
        return len(self.images)

    def __getitem__(self, img_id):
        return self.images[img_id]

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_idx >= len(self.images):
            raise StopIteration
        cur = self.current_idx
        self.current_idx += 1
        return self.images[cur]

    def add_image(self, img_id: int, absolute_path: Path):
        new_image = Image(img_id, absolute_path)
        self.images.append(new_image)

    @property
    def img_names(self):
        return [im.name for im in self.images]

    @property
    def img_paths(self):
        return [im.absolute_path for im in self.images]
