call ".\env\scripts\activate"
python test_gen_images.py --dataset vimeo90K_septuplet --data_root ./vimeo_septuplet --load_from "D:\SUTD\Term 7\CV\T7CV\saved_models_final\vimeo90K_septuplet\Flavr_w_inception_net\Flavr_w_inception_encoder_model_best.pth" --n_outputs 1 --exp_name Flavr_w_inception_encoder_model_best --num_workers 2