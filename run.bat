@echo off
REM python ./main.py -i assets/imgs -o assets/outs -m custom_pairs -f loftr -n 8000 -p ./assets/pairs.txt
REM python ./main.py -i assets/imgs -o assets/outs -m sequential -f loftr -n 8000 -v 1
REM python ./main.py -i assets/imgs -o assets/outs -m sequential -f detect_and_describe -n 8000 -v 1
python ./main.py -i C:\Users\lmorelli\Desktop\Luca\GitHub_3DOM\Ferox\cam0\data -o assets/outs -m custom_pairs -f superglue -n 8000 -p C:\Users\lmorelli\Desktop\Luca\GitHub_3DOM\Ferox\pairs3.txt
