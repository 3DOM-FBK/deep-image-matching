# python ./main.py --help

# SEQUENTIAL
# python ./main.py  --config superpoint+lightglue --images assets/example_cyprus --outs assets/output --strategy sequential --overlap 1 --force
# python ./main.py  --config superpoint+lightglue --images assets/example_cyprus --outs assets/output --strategy sequential --overlap 1 --force

### BRUTEFORCE

# example_cyprus
# python ./main.py  --config superpoint+lightglue               --images assets/example_cyprus --outs assets/output --strategy bruteforce --force --skip_reconstruction 
# python ./main.py  --config superpoint+lightglue               --images assets/example_cyprus --outs assets/output --strategy bruteforce --force --skip_reconstruction --upright
# python ./main.py  --config superpoint+lightglue_fast          --images assets/example_cyprus --outs assets/output --strategy bruteforce --force --skip_reconstruction
# python ./main.py  --config superpoint+superglue               --images assets/example_cyprus --outs assets/output --strategy bruteforce --force --skip_reconstruction
# python ./main.py  --config disk+lightglue                     --images assets/example_cyprus --outs assets/output --strategy bruteforce --force --skip_reconstruction
# python ./main.py  --config aliked+lightglue                   --images assets/example_cyprus --outs assets/output --strategy bruteforce --force --skip_reconstruction
# python ./main.py  --config orb+kornia_matcher                 --images assets/example_cyprus --outs assets/output --strategy bruteforce --force --skip_reconstruction
# python ./main.py  --config sift+kornia_matcher                --images assets/example_cyprus --outs assets/output --strategy bruteforce --force --skip_reconstruction
# python ./main.py  --config keynetaffnethardnet+kornia_matcher --images assets/example_cyprus --outs assets/output --strategy bruteforce --force --skip_reconstruction
# python ./main.py  --config dedode                             --images assets/example_cyprus --outs assets/output --strategy bruteforce --force --skip_reconstruction
# python ./main.py  --config roma                               --images assets/example_cyprus --outs assets/output --strategy bruteforce --force --skip_reconstruction
# python ./main.py  --config loftr                              --images assets/example_cyprus --outs assets/output --strategy bruteforce --force --skip_reconstruction

# example_sacre_coeur
# python ./main.py  --config loftr    --images assets/example_sacre_coeur --outs assets/output --strategy bruteforce --force
# python ./main.py  --config se2loftr --images assets/example_sacre_coeur --outs assets/output --strategy bruteforce --force
# python ./main.py  --config roma     --images assets/example_sacre_coeur --outs assets/output --strategy bruteforce --force
# python ./main.py  --config se2loftr --images assets/example_sacre_coeur --outs assets/output --strategy bruteforce --force


# examples
# python ./main.py  --config superpoint+lightglue               --images assets/example_square       --outs assets/output --strategy bruteforce --force
# python ./main.py  --config loftr                              --images assets/example_easy_highres --outs assets/output --strategy bruteforce --force
# python ./main.py  --config loftr                              --images assets/example_temple_bal   --outs assets/output --strategy bruteforce --force
# python ./main.py  --config roma                              --images assets/example_temple_bal   --outs assets/output --strategy bruteforce --force
# python ./main.py  --config roma                               --images assets/example_temple_bal   --outs assets/output --strategy retrieval -r cosplace --force
# python ./main.py  --config roma                               --images assets/example_square       --outs assets/output --strategy retrieval -r openibl  --force