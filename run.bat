# SEQUENTIAL
python ./main.py --config superpoint+lightglue --dir assets/example_cyprus --strategy sequential --overlap 1 --force
python ./main.py --config superpoint+lightglue --dir assets/example_cyprus --strategy sequential --overlap 1 --force

### BRUTEFORCE

# example_cyprus
python ./main.py --config superpoint+lightglue --dir assets/example_cyprus --strategy bruteforce --force --skip_reconstruction
python ./main.py --config superpoint+lightglue --dir assets/example_cyprus --strategy bruteforce --force --skip_reconstruction --upright
python ./main.py --config superpoint+lightglue_fast --dir assets/example_cyprus --strategy bruteforce --force --skip_reconstruction
python ./main.py --config superpoint+superglue --dir assets/example_cyprus --strategy bruteforce --force --skip_reconstruction
python ./main.py --config disk+lightglue --dir assets/example_cyprus --strategy bruteforce --force --skip_reconstruction
python ./main.py --config aliked+lightglue --dir assets/example_cyprus --strategy bruteforce --force --skip_reconstruction
python ./main.py --config orb+kornia_matcher --dir assets/example_cyprus --strategy bruteforce --force --skip_reconstruction
python ./main.py --config sift+kornia_matcher --dir assets/example_cyprus --strategy bruteforce --force --skip_reconstruction
python ./main.py --config keynetaffnethardnet+kornia_matcher --dir assets/example_cyprus --strategy bruteforce --force --skip_reconstruction
python ./main.py --config dedode --dir assets/example_cyprus --strategy bruteforce --force --skip_reconstruction
python ./main.py --config roma --dir assets/example_cyprus --strategy bruteforce --force --skip_reconstruction
python ./main.py --config loftr --dir assets/example_cyprus --strategy bruteforce --force --skip_reconstruction

# example_sacre_coeur
python ./main.py --config loftr --dir assets/example_sacre_coeur --strategy bruteforce --force
python ./main.py --config se2loftr --dir assets/example_sacre_coeur --strategy bruteforce --force
python ./main.py --config roma --dir assets/example_sacre_coeur --strategy bruteforce --force
python ./main.py --config se2loftr --dir assets/example_sacre_coeur --strategy bruteforce --force

# examples
python ./main.py --config superpoint+lightglue --dir assets/example_square --strategy bruteforce --force
python ./main.py --config loftr --dir assets/example_easy_highres --strategy bruteforce --force
python ./main.py --config loftr --dir assets/example_temple_bal --strategy bruteforce --force
python ./main.py --config roma --dir assets/example_temple_bal --strategy bruteforce --force
python ./main.py --config roma --dir assets/example_temple_bal --strategy retrieval -r cosplace --force
python ./main.py --config roma --dir assets/example_square --strategy retrieval -r openibl --force
