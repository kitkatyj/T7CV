cd /d "D:\SUTD\Term 7\CV\T7CV"
call activate flavr
python main.py --batch_size 8 --test_batch_size 8 --dataset vimeo90K_septuplet --loss 1*L1 --max_epoch 200 --lr 0.0002 --data_root "D:\SUTD\Term 7\CV\T7CV\vimeo_septuplet" --n_outputs 1 --exp_name "Flavr_w_inception_net_v2_flavr_env"