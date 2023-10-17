@echo off
REM python ./main.py -i assets/imgs -o assets/outs -m custom_pairs -f loftr -n 8000 -p ./assets/pairs.txt
python ./main.py -i assets/imgs -o assets/outs -m sequential -f loftr -n 8000 -v 1
REM python ./main.py -i assets/imgs -o assets/outs -m sequential -f detect_and_describe -n 8000 -v 1