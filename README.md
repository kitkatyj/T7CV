# FLAVR-Based Video Frame Interpolation Methods

## Overview
This repository contains research and implementation for video frame interpolation using the [FLAVR (Flow-Agnostic Video Representations)](https://tarun005.github.io/FLAVR/) network. Our project extends the capabilities of FLAVR by experimenting with several architectural enhancements and evaluating their performance in generating interpolated video frames.

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

# Usage
https://sutdapac-my.sharepoint.com/personal/joshua_limhongjun_mymail_sutd_edu_sg/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fjoshua%5Flimhongjun%5Fmymail%5Fsutd%5Fedu%5Fsg%2FDocuments%2FT7%5FCV%5FCheckpoints&ga=1

## Conclusion
Our extensive evaluations suggest that integrating ConvLSTM into the FLAVR architecture enhances interpolation quality. Moreover, the simplicity and efficiency of the Single Frame model present it as a viable alternative for less computationally intense applications.


