# Example
# python ./main.py -i assets/imgs -o assets/outs -m retreival -r netvlad

import os
import argparse

from pathlib import Path

from lib.match_imgs import MatchImgs

def main():
    parser = argparse.ArgumentParser(description="Matching with hand-crafted and deep-learning based local features and image retrieval.")
    
    parser.add_argument('-i', '--images', type=str, help="Input image folder")
    parser.add_argument('-o', '--outs', type=str, help="Output folder")
    parser.add_argument('-m', '--matcher', choices=['bruteforce', 'sequential', 'retreival'])
    parser.add_argument('-r', '--retrieval', choices=['netvlad', 'openibl', 'cosplace', 'dir'])
    #parser.add_argument('-v', '--verbose', action='store_true', help="Enable verbose mode")
    
    args = parser.parse_args()

    imgs_dir = Path(args.images)
    output_dir = Path(args.outs)
    retrieval_option = args.retrieval
    matching_option = args.matcher
    
    MatchImgs(matching_option, imgs_dir, output_dir, retrieval_option)
    

if __name__ == "__main__":
    main()