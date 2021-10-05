# W18 RH
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port '26001' tools/train_ee.py --cfg experiments/cityscapes/w18.yaml OUTPUT_DIR output_new/w18/RH/    MODEL.NAME model_anytime   MODEL.EXTRA.EE_WEIGHTS '(1,1,1,1)' EXIT.TYPE 'downup_pool_1x1_inter_flexible' EXIT.FIX_INTER_CHANNEL True  EXIT.INTER_CHANNEL 64 TRAIN.END_EPOCH 484 MASK.FULL_USE True    WORKERS 4

# W18 SA
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port '26001' tools/train_ee.py --cfg experiments/cityscapes/w18.yaml OUTPUT_DIR output_new/w18/SA    MODEL.NAME model_anytime   MODEL.EXTRA.EE_WEIGHTS '(1,1,1,1)'  TRAIN.END_EPOCH 484 MASK.FULL_USE True   MASK.CONF_THRE 0.998   WORKERS 4

# W18 EE
 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port '26001' tools/train_ee.py --cfg experiments/cityscapes/w18.yaml OUTPUT_DIR output_new/w18/EE    MODEL.NAME model_anytime   MODEL.EXTRA.EE_WEIGHTS '(1,1,1,1)'  TRAIN.END_EPOCH 484

# W18 FULL
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port '26001' tools/train_ee.py --cfg experiments/cityscapes/w18.yaml OUTPUT_DIR output_new/w18/FULL    MODEL.NAME model_anytime   MODEL.EXTRA.EE_WEIGHTS '(1,1,1,1)'  TRAIN.END_EPOCH 484 EXIT.TYPE 'downup_pool_1x1_inter_flexible' EXIT.FIX_INTER_CHANNEL True  EXIT.INTER_CHANNEL 64 MASK.FULL_USE True   MASK.CONF_THRE 0.998


# W48 RH
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port '26001' tools/train_ee.py --cfg experiments/cityscapes/w48.yaml OUTPUT_DIR output_new/w48/RH    MODEL.NAME model_anytime   MODEL.EXTRA.EE_WEIGHTS '(1,1,1,1)' EXIT.TYPE 'downup_pool_1x1_inter_flexible' EXIT.FIX_INTER_CHANNEL True  EXIT.INTER_CHANNEL 64 TRAIN.END_EPOCH 484 MASK.FULL_USE True    WORKERS 4

# W48 SA
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port '26001' tools/train_ee.py --cfg experiments/cityscapes/w48.yaml OUTPUT_DIR output_new/w48/SA    MODEL.NAME model_anytime   MODEL.EXTRA.EE_WEIGHTS '(1,1,1,1)'  TRAIN.END_EPOCH 484 MASK.FULL_USE True   MASK.CONF_THRE 0.998   WORKERS 4

# W48 EE
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port '26001' tools/train_ee.py --cfg experiments/cityscapes/w48.yaml OUTPUT_DIR output_new/w48/EE    MODEL.NAME model_anytime   MODEL.EXTRA.EE_WEIGHTS '(1,1,1,1)'  TRAIN.END_EPOCH 484

# W48 FULL
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port '26001' tools/train_ee.py --cfg experiments/cityscapes/w48.yaml OUTPUT_DIR output_new/w48/FULL    MODEL.NAME model_anytime   MODEL.EXTRA.EE_WEIGHTS '(1,1,1,1)'  TRAIN.END_EPOCH 484 EXIT.TYPE 'downup_pool_1x1_inter_flexible' EXIT.FIX_INTER_CHANNEL True  EXIT.INTER_CHANNEL 64 MASK.FULL_USE True   MASK.CONF_THRE 0.998