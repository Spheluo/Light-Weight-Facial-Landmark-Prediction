# Light-Weight-Facial-Landmark-Prediction
ShuffleNet V2 with heatmapped landmarks output

## Introduction
We aim to build a model to predict 68 2D facial landmarks in a single cropped face image with high accuracy, high efficiency, and low computational costs.
![image](https://github.com/user-attachments/assets/fd6983de-bec9-4716-ab8c-110f6e88789e)

## Applications
Applications including "Facial Motion Retargrting" and "Talking Head Generation"
![image](https://github.com/user-attachments/assets/995111e3-f15c-43d2-a354-5cb8eeaf42f7)

## Training Data
100k diverse synthetic facial images with 68 2D landmark coordinates
![image](https://github.com/user-attachments/assets/5794dcc4-40b8-4607-8acf-3e0fc6d98e44)
E. Wood et al., “Fake it till you make it: face analysis in the wild using synthetic data alone”, in ICCV 2021

## Validation/Testing Datasets - AFLW2000-3D
The fitted 3D faces of the first 2000 AFLW samples, which can be used for 3D face alignment evaluation
![image](https://github.com/user-attachments/assets/7e8ddd81-dbac-4537-ad00-dd0e296e5741)
X. Zhu et al., “Face Alignment Across Large Poses: A 3D Solution”, in CVPR 2016

## Protocol and Constraints
1. We randomly split the AFLW2000 dataset into 10% and 90% as the validation and testing dataset in this challenge
2. Face Detector is not allowed during training
3. The upper bound of model size is 15MB
4. The model should contain weights (such as neural network)
5. We target on float32 solutions. (Float16, int8 or any other quantization methods are not allowed.)
6. You can ONLY train a single model. You cannot train an ensembled model.

## Evaluation Metric
Under model constraints, we evaluate the results on the testing dataset with the commonly used Normalized mean error (NME). 
For a matched prediction and the bounding box with width W and height H, we calculate the NME as follows,
![image](https://github.com/user-attachments/assets/49ff1aad-679b-4634-9459-fd68042a9cc0)

## Dataset format
Download link : 
https://drive.google.com/file/d/1hhcsXxGehgf_wf2QJKSuwB7e3xxrTYn9/view?usp=sharing
![image](https://github.com/user-attachments/assets/65c03d12-3a38-4fe3-b9be-c67fbfa462af)
![image](https://github.com/user-attachments/assets/51a3b91a-eccd-4df3-afbe-d75e17891bf6)

## How to calculate the NME by yourself
![image](https://github.com/user-attachments/assets/097c9411-5b19-45a6-ba63-7928dc374380)

## Testing Dataset
Testing dataset link : https://drive.google.com/file/d/1KNkLYqiZtqeftt-1PQmirzriwmgHI1-/view?usp=sharing

