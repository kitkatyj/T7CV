call ".\venv\scripts\activate"
py test.py --batch_size 8 --test_batch_size 8 --dataset vimeo90K_septuplet --loss 1*L1 --max_epoch 200 --lr 0.0002 --data_root "C:\vimeo_septuplet" --n_outputs 1 --exp_name "v2_w_inception"

python test.py --dataset vimeo90K_septuplet --data_root "C:\vimeo_septuplet" --load_from <saved_model> --n_outputs 1