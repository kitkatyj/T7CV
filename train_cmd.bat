@echo off
rem Training command
python main.py --batch_size 32 --test_batch_size 32 --dataset vimeo90K_septuplet --loss 1*L1 --max_epoch 200 --lr 0.00002 --data_root ./vimeo_septuplet --n_outputs 1 --exp_name SingleFrame_stack-AE-fullVGG --checkpoint_dir ckpts --num_workers 4