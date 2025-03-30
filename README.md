# Smart-Glove

For more details and files, please see [https://jacoo-ai.github.io/Smart-Glove/](https://)

## Project Overview

This project addresses the challenge of real-time hand gesture prediction to enable engaging, rhythm-based interactive experiences. Specifically, we introduce Fist Dance, a music-driven game that integrates gestures inspired by "Rock, Paper, Scissors" with directional movements. To support this gameplay, we developed a wearable gesture prediction system designed for robustness across users and hardware variations. The system integrates five flex sensors and one IMU on a glove to capture both finger movements and hand orientation. Over 80,000 samples were collected from 10 participants and labeled using a beat-aligned sliding window method, covering 24 unique gesture transitions. A three-layer LSTM model was trained in a three-stage pipeline: large-scale pretraining, user-specific finetuning, and hardware adaptation. The final model achieved up to 93 $\%$ accuracy and 93.9$\%$ precision, demonstrating high performance even after sensor replacements. This work showcases the effective combination of wearable sensing and sequential deep learning for building engaging and adaptive human-computer interaction systems.

## Dataset

The raw data is in `sensor_data` and the labeled data is in `dataset` .
