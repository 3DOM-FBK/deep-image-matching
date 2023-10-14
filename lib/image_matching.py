from pathlib import Path
from lib.pairs_generator import PairsGenerator
from lib.image_list import ImageList

class ImageMatching:
    def __init__(
            self, 
            imgs_dir : Path, 
            matching_strategy : str, 
            pair_file : Path, 
            retrieval_option : str,
            overlap : int,
            ):
        
        self.image_list = ImageList(imgs_dir)
        images = self.image_list.img_names
 
        if len(images) == 0:
            raise ValueError("Image folder empty. Supported formats: '.jpg', '.JPG', '.png'")
        elif len(images) == 1:
            raise ValueError("Image folder must contain at least two images")

        self.pairs = []
        if pair_file is not None and matching_strategy == 'custom_pairs':
            with open(pair_file, 'r') as txt_file:
                lines = txt_file.readlines()
                for line in lines:
                    im1, im2 = line.strip().split(' ', 1)
                    self.pairs.append((im1, im2))
        else:
            pairs_generator = PairsGenerator(self.image_list.img_paths, matching_strategy, retrieval_option, overlap)
            self.pairs = pairs_generator.run()

        self.keypoints = {}
        self.correspondences = {}

    def run(self):
        return self.image_list.img_names, self.pairs, self.keypoints, self.correspondences