# default training
nohup python /home/yepeng_liu/code_python/LiftFeat/train.py \
--name LiftFeat_test \
--ckpt_save_path /home/yepeng_liu/code_python/LiftFeat/trained_weights/test \
--device_num 1 \
--use_megadepth \
--megadepth_batch_size 8 \
--use_coco \
--coco_batch_size 4 \
--save_ckpt_every 1000 \
> /home/yepeng_liu/code_python/LiftFeat/trained_weights/test/training.log 2>&1 &