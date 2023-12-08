@echo off
REM python ./main.py  --config superpoint+lightglue --images assets/example_cyprus --outs assets/output --strategy sequential --overlap 1
REM python ./main.py  --config superpoint+lightglue --images assets/example_cyprus --outs assets/output --strategy sequential --overlap 1
REM python ./main.py  --config superpoint+lightglue --images assets/example_cyprus --outs assets/output --strategy bruteforce --force
REM python ./main.py  --config superpoint+lightglue_fast --images assets/example_cyprus --outs assets/output --strategy bruteforce --force
python ./main.py  --config superpoint+superglue --images assets/example_cyprus --outs assets/output --strategy bruteforce --force
REM python ./main.py  --config disk+lightglue --images assets/example_cyprus --outs assets/output --strategy bruteforce --force
REM python ./main.py  --config aliked+lightglue --images assets/example_cyprus --outs assets/output --strategy bruteforce --force
REM python ./main.py  --config orb+kornia_matcher --images assets/example_cyprus --outs assets/output --strategy bruteforce --force
REM python ./main.py  --config sift+kornia_matcher --images assets/example_cyprus --outs assets/output --strategy bruteforce --force
REM python ./main.py  --config keynetaffnethardnet+kornia_matcher --images assets/example_cyprus --outs assets/output --strategy bruteforce --force