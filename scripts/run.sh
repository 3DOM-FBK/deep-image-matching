# SEQUENTIAL
#python ./main.py --pipeline superpoint+lightglue --dir assets/pytest --strategy sequential --skip_reconstruction --overlap 1 --force
#python ./main.py --pipeline superpoint+lightglue --dir assets/pytest --strategy sequential --overlap 1 --force

# BRUTEFORCE
#python ./main.py --pipeline superpoint+lightglue --dir assets/pytest --strategy bruteforce --force --skip_reconstruction --config ./config/superpoint+lightglue.yaml
#python ./main.py --pipeline superpoint+lightglue --dir assets/pytest --strategy bruteforce --force --upright
#python ./main.py --pipeline superpoint+lightglue_fast --dir assets/pytest --strategy bruteforce --force --skip_reconstruction
#python ./main.py --pipeline superpoint+superglue --dir assets/pytest --strategy bruteforce --force --skip_reconstruction --config ./config/superpoint+superglue.yaml
#python ./main.py --pipeline disk+lightglue --dir assets/pytest --strategy bruteforce --force --skip_reconstruction
#python ./main.py --pipeline aliked+lightglue --dir assets/pytest --strategy bruteforce --force --skip_reconstruction
#python ./main.py --pipeline orb+kornia_matcher --dir assets/pytest --strategy bruteforce --force
#python ./main.py --pipeline sift+kornia_matcher --dir assets/pytest --strategy bruteforce --force --skip_reconstruction --config ./config/sift.yaml
#python ./main.py --pipeline keynetaffnethardnet+kornia_matcher --dir assets/pytest --strategy bruteforce --force
##python ./main.py --pipeline dedode --dir assets/pytest --strategy bruteforce --force --skip_reconstruction
#python ./main.py --pipeline roma --dir assets/pytest --strategy bruteforce --force --skip_reconstruction --quality high --tiling none -V --config ./config/roma.yaml
#python ./main.py --pipeline roma --dir assets/example_temple_baal --strategy bruteforce --force --skip_reconstruction --quality high --tiling none -V
#python ./main.py --pipeline loftr --dir assets/pytest --strategy bruteforce --force --skip_reconstruction

## EXAMPLE SACRE COEUR
#python ./main.py --pipeline loftr --dir assets/example_sacre_coeur --strategy bruteforce --force
#python ./main.py --pipeline se2loftr --dir assets/example_sacre_coeur --strategy bruteforce --force
#python ./main.py --pipeline roma --dir assets/example_sacre_coeur --strategy bruteforce --force
#python ./main.py --pipeline se2loftr --dir assets/example_sacre_coeur --strategy bruteforce --force

## OTHER EXAMPLES
#python ./main.py --pipeline superpoint+lightglue --dir assets/example_square --strategy bruteforce --force
#python ./main.py --pipeline loftr --dir assets/example_easy_highres --strategy bruteforce --force
#python ./main.py --pipeline loftr --dir assets/example_temple_bal --strategy bruteforce --force
#python ./main.py --pipeline roma --dir assets/example_temple_bal --strategy bruteforce --force
#python ./main.py --pipeline roma --dir assets/example_temple_bal --strategy retrieval -r cosplace --force
#python ./main.py --pipeline roma --dir assets/example_square --strategy retrieval -r openibl --force