@echo off
rem Training command
python main.py --batch_size 16 --test_batch_size 16 --dataset vimeo90K_septuplet --loss 1*L1 --max_epoch 200 --lr 0.00002 --data_root ./vimeo_septuplet --n_outputs 1 --exp_name SingleFrame_stack-Unet-VGG16 --checkpoint_dir ckpts --num_workers 2