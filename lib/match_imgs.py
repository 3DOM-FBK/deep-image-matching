from lib.image_retrieval import ImageRetrieval

def MatchImgs(matcher_option, imgs_dir, output_dir, retrieval_option):
    img_pairs = []

    if matcher_option == 'retreival':
        ImageRetrieval(imgs_dir, output_dir, retrieval_option)
    else:
        print('Matching approch not implmented yet. Quit')
        quit()
    
    with open(output_dir / 'pairs.txt', 'r') as pairs:
        lines = pairs.readlines()
        for line in lines:
            im1, im2 = line.strip().split(' ', 1)
            img_pairs.append((im1, im2))
    
    index_duplicate_pairs = []
    for i in range(len(img_pairs)-1):
        pair1 = img_pairs[i]
        im1 = pair1[0]
        im2 = pair1[1]
        for j in range(i+1, len(img_pairs)):
            pair2 = img_pairs[j]
            im3 = pair2[0]
            im4 = pair2[1]
            if im3 == im1 and im4 == im2:
                index_duplicate_pairs.append(j)
                print('discarded', im1, im2, im3, im4)
            elif im3 == im2 and im4 == im1:
                index_duplicate_pairs.append(j)
                print('discarded', im1, im2, im3, im4)
            else:
                pass

    with open(output_dir / 'pairs_no_duplicates.txt', 'w') as final_pairs:
        for i in range(len(img_pairs)-1):
            if i not in index_duplicate_pairs:
                final_pairs.write(f"{img_pairs[i][0]} {img_pairs[i][1]}\n")
    

