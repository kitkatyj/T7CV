# FLAVR-Based Video Frame Interpolation Methods

## Overview
This repository contains research and implementation for video frame interpolation using the [FLAVR (Flow-Agnostic Video Representations)](https://tarun005.github.io/FLAVR/) network. Our project extends the capabilities of FLAVR by experimenting with several architectural enhancements and evaluating their performance in generating interpolated video frames. This is a SUTD university project for the module "50.035 Computer Vision".

## Team

| Name                | Student ID |
|---------------------|------------|
| Lee Pei Xuan        | 1005513    |
| Pala Tej Deep       | 1005282    |
| Lim Hong Jun, Joshua| 1005259    |
| Baarath S/O Sellathurai | 1005433    |
| Kat Yong Jie        | 1005167    |

## Introduction
Video frame interpolation is crucial for applications such as slow-motion video generation, frame rate conversion, and more. We've built upon the FLAVR architecture to explore its adaptability and efficacy under various conditions including occlusions and dynamic lighting.

## Methodology
Our methodology encompasses the following key components:

- **Evaluation Metrics:** We used PSNR and SSIM indices to quantify model performances, with higher values indicating better quality interpolations.
- **Datasets:** Training was conducted using the Vimeo-90K dataset, complemented by our custom test dataset for additional analysis.
- **Training Parameters:** We standardized training across models using the Adam optimizer, focusing on L1 loss after comparing it with other loss functions.

## Summary of Findings
We experimented with six model architectures, including variations of FLAVR, UNETr, and models incorporating ConvLSTM and Inception networks. Below is a summary table of our key findings:

| Model Type                   | Best PSNR | Best SSIM |
|------------------------------|-----------|-----------|
| Original FLAVR               | 18.275    | 0.687     |
| FLAVR w/ Inception           | 18.851    | 0.676     |
| FLAVR w/ ConvLSTM            | 21.597    | 0.590     |
| UNETr                        | 15.813    | 0.325     |
| FlowNet                      | 14.939    | 0.418     |
| Masked Image Modelling       | 16.162    | 0.487     |
| Single Frame (2D CNN)        | 18.508    | 0.652     |

The FLAVR model with ConvLSTM showed the most significant improvement in PSNR, indicating its strong capacity to interpolate frames accurately.

## Training model on Vimeo-90K septuplets

For training your own model on the Vimeo-90K dataset, use the following command. You can download the original dataset from [this link](http://toflow.csail.mit.edu/).
``` bash
python main.py --batch_size 32 --test_batch_size 32 --dataset vimeo90K_septuplet --model <model_name> --loss 1*L1 --max_epoch 200 --lr 0.0002 --data_root <dataset_path> --n_outputs 1
```
You can run main.py with the following models and loss function
```bash
--model: ["singleframe", "mim", "opticalflow","flavr_inception", "flavr_lstm", "unetr"]
--loss: weight*loss_types+...
loss_types: ['MSE','Huber','L1','VGG22', 'VGG33', 'VGG44', 'VGG54','VGGP','SSIM', 'GAN']
```

## Pretrained Checkpoint Usage

To further enhance the training process, you can utilize the checkpoints from our best-performing models. 
You can download the checkpoints from our OneDrive repository:
[Download Checkpoints](https://sutdapac-my.sharepoint.com/personal/joshua_limhongjun_mymail_sutd_edu_sg/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fjoshua%5Flimhongjun%5Fmymail%5Fsutd%5Fedu%5Fsg%2FDocuments%2FT7%5FCV%5FCheckpoints&ga=1)

To use a checkpoint in your training, add the `--load_from` flag to your training command, followed by the path to the downloaded checkpoint file:

```bash
python main.py --batch_size 32 --test_batch_size 32 --dataset vimeo90K_septuplet --model <model_name> --loss 1*L1 --max_epoch 200 --lr 0.0002 --data_root <dataset_path> --n_outputs 1 --load_from <path_to_checkpoint>
```

## Testing using trained model

### Trained Models
You can download the pretrained original FLAVR models from the following links.
 Method        | Trained Model  |
| ------------- |:-----|
| **2x** | [Link](https://drive.google.com/drive/folders/1M6ec7t59exOSlx_Wp6K9_njBlLH2IPBC?usp=sharing) |
| **4x** |   [Link](https://drive.google.com/file/d/1btmNm4LkHVO9gjAaKKN9CXf5vP7h4hCy/view?usp=sharing)   |
| **8x** |   [Link](https://drive.google.com/drive/folders/1Gd2l69j7UC1Zua7StbUNcomAAhmE-xFb?usp=sharing)  |

For testing a pretrained model on Vimeo-90K septuplet validation set, you can run the following command:
```bash
python test.py --dataset vimeo90K_septuplet --model <model_name> --data_root <data_path> --load_from <saved_model> --n_outputs 1
```

## Testing using Custom Dataset

### Download the Dataset
Our custom dataset for testing, can be downloaded from our OneDrive repository:
[Download Custom Occlusion Dataset](https://sutdapac-my.sharepoint.com/personal/joshua_limhongjun_mymail_sutd_edu_sg/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fjoshua%5Flimhongjun%5Fmymail%5Fsutd%5Fedu%5Fsg%2FDocuments%2FT7%5FCV%5FCheckpoints%2FCustom%20Occlusion%20Dataset)

### Steps for Running Test on Custom Dataset
1. Place the `sep_testlist_occ` file in the `vimeo_septuplet/` directory.
2. Unzip `occlusion.zip` and move the `occlusion` folder into `vimeo_septuplet/sequences`.
3. Modify the dataset script `vimeo90k_septuplet.py` located in `T7CV/dataset/` by changing the following:
    - On line 24, replace the path with `test_fn = os.path.join(self.data_root, 'sep_testlist_occ.txt')`.
    - Ensure `sep_testlist_occ.txt` is used instead of the original `sep_testlist.txt`.

After these setup steps, execute the `test_gen_images.py` script to run inference on the occlusion samples.



## Conclusion
Our extensive evaluations suggest that integrating ConvLSTM into the FLAVR architecture enhances interpolation quality. Moreover, the simplicity and efficiency of the Single Frame model present it as a viable alternative for less computationally intense applications.


