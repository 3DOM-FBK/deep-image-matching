import os
import shutil
import argparse

from pathlib import Path
from lib.export_to_colmap import ExportToColmap
from lib.image_matching import ImageMatching
from lib.deep_image_matcher.logger import setup_logger
#from lib.match_imgs import MatchImgs


def main():
    setup_logger()

    parser = argparse.ArgumentParser(description="Matching with hand-crafted and deep-learning based local features and image retrieval.")
    parser.add_argument('-i', '--images', type=str, help="Input image folder", required=True)
    parser.add_argument('-o', '--outs', type=str, help="Output folder", required=True)
    parser.add_argument('-m', '--strategy', choices=['bruteforce', 'sequential', 'retrieval', 'custom_pairs'], required=True)
    parser.add_argument('-p', '--pairs', type=str)
    parser.add_argument('-r', '--retrieval', choices=['netvlad', 'openibl', 'cosplace', 'dir'])
    parser.add_argument('-v', '--overlap', type=int, help="Image overlap if using sequential overlap strategy")
    #parser.add_argument('-v', '--verbose', action='store_true', help="Enable verbose mode")
    args = parser.parse_args()

    if args.strategy == "retrieval" and args.retrieval is None:
        parser.error("--retrieval option is required when --strategy is set to retrieval")
    elif args.strategy == "retrieval":
        retrieval_option = args.retrieval
    else: 
        retrieval_option = None

    if args.strategy == "custom_pairs" and args.pairs is None:
        parser.error("--pairs option is required when --strategy is set to custom_pairs")
    elif args.strategy == "custom_pairs":
        pair_file = Path(args.pairs)
    else:
        pair_file = None

    if args.strategy == "sequential" and args.overlap is None:
        parser.error("--overlap option is required when --strategy is set to sequential")
    elif args.strategy == "sequential":
        overlap = args.overlap
    else:
        overlap = None

    imgs_dir = Path(args.images)
    output_dir = Path(args.outs)
    matching_strategy = args.strategy
    
    if output_dir.exists() and output_dir.is_dir():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run matching
    img_matching = ImageMatching(imgs_dir, matching_strategy, pair_file, retrieval_option, overlap)
    images = img_matching.img_names()
    pairs =  img_matching.generate_pairs()
    keypoints, correspondences = img_matching.match_pairs()

    # Plot statistics
    print("\n Finished matching and exporting")
    print("n processed images: ", len(images))
    print("n processed pairs: ", len(pairs))

    # Export in colmap format
    ExportToColmap(keypoints, correspondences, output_dir)
    

if __name__ == "__main__":
    main()