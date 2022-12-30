
echo Evaluate Audio-Visual Model
python main.py -c configs/LRS23/AV/EffConfInterCTC.py --mode evaluation --checkpoint checkpoints_swa-equal-60-70.ckpt
echo

echo Evaluate Visual-Only Model
python main.py -c configs/LRS23/VO/EffConfInterCTC.py --mode evaluation --checkpoint checkpoints_swa-equal-90-100.ckpt
echo

echo Evaluate Audio-Only Model
python main.py -c configs/LRS23/AO/EffConfCTC.py --mode evaluation --checkpoint checkpoints_swa-equal-200-210.ckpt
echo

echo Evaluate LM
python main.py -c configs/LRS23/LM/GPT-Small.py --mode evaluation --load_last
echo

echo Evaluate LRW
python main.py -c configs/LRW/EffConfCE.py --mode evaluation --load_last
echo