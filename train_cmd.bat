@echo off
rem Training command
python main.py --batch_size 8 --test_batch_size 8 --dataset vimeo90K_septuplet --loss 1*L1 --max_epoch 200 --lr 0.00002 --data_root ./vimeo_septuplet --n_outputs 1 --exp_name MIM_Transformer --checkpoint_dir ckpts --num_workers 2
@REM python test.py --dataset vimeo90K_septuplet --data_root ./vimeo_septuplet --n_outputs 1 --load_from ./ckpts\saved_models_final\vimeo90K_septuplet\MIM_Transformer\model_best.pth --num_workers 2

@REM python test_gen_images.py --dataset vimeo90K_septuplet --data_root ./vimeo_septuplet --load_from ckpts\saved_models_final\vimeo90K_septuplet\FlownetS-pre-bn-LAE\model_best.pth --n_outputs 1 --exp_name FlownetS-pre-bn-LAE