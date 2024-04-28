# VQP-Net

1. Resnet_distortion_classification_and_ranking for pretraining with augmented data

## Transfer learning:
1. First run Resnet_frame_level_score.py to generate frame level scores
2. Then run video_score_fcnn.py to get video scores

## End-to-end learning:

Run end_to_end_VQPnet.py. Other configurations besides FCNN can also be run by 
uncommenting specific parts of code. Other configurations include LSTM, RNN and GRU. 

## Usage

If you intend to use this method or code in your work, please cite the following paper:

**Khan, Z. A., Beghdadi, A., Kaaniche, M., Alaya-Cheikh, F., & Gharbi, O. (2022). A neural network based framework for effective laparoscopic video quality assessment. Computerized Medical Imaging and Graphics, 101, 102121.

and in case you also use the referenced LVQ Database, please cite the following paper:

** Khan, Z.A., Beghdadi, A., Cheikh, F.A., Kaaniche, M., Pelanis, E., Palomar, R., Fretland, Å.A., Edwin, B. and Elle, O.J., 2020, March. "Towards a video quality assessment based framework for enhancement of laparoscopic videos". In Medical Imaging 2020: Image Perception, Observer Performance, and Technology Assessment (Vol. 11316, p. 113160P). International Society for Optics and Photonics.
