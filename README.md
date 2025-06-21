# Light-Weight-Facial-Landmark-Prediction
ShuffleNet V2 with heatmapped landmarks output

## Introduction
We aim to build a model to predict 68 2D facial landmarks in a single cropped face image with high accuracy, high efficiency, and low computational costs.  
![image](https://github.com/user-attachments/assets/fd6983de-bec9-4716-ab8c-110f6e88789e)

## Applications
Applications including "Facial Motion Retargrting" and "Talking Head Generation"  
![image](https://github.com/user-attachments/assets/995111e3-f15c-43d2-a354-5cb8eeaf42f7)  

Through literature review, we initially examined a paper published at ICCV 2017 that employed the **Face Alignment Network (FAN)** to train a facial recognition model [1]. As illustrated below, this model stacks four Hourglass Networks (HG) and uses a **heatmap-based** approach to locate key facial landmarks. However, this model is highly complex and results in an overly large model size. Therefore, we explored alternative options and ultimately selected **ShuffleNet V2** as our training model due to its efficiency and lightweight nature.  

![image](https://github.com/user-attachments/assets/8a190d13-b124-4db5-a9ec-5eb2b1867790)
## Training Data
100k diverse synthetic facial images with 68 2D landmark coordinates
![image](https://github.com/user-attachments/assets/5794dcc4-40b8-4607-8acf-3e0fc6d98e44)
E. Wood et al., “Fake it till you make it: face analysis in the wild using synthetic data alone”, in ICCV 2021

## Validation/Testing Datasets - AFLW2000-3D
The fitted 3D faces of the first 2000 AFLW samples, which can be used for 3D face alignment evaluation  
![image](https://github.com/user-attachments/assets/7e8ddd81-dbac-4537-ad00-dd0e296e5741)  
Testing dataset link : https://drive.google.com/file/d/1KNkLYqiZtqeftt-1PQmirzriwmgHI1-/view?usp=sharing  
X. Zhu et al., “Face Alignment Across Large Poses: A 3D Solution”, in CVPR 2016  
  
## Dataset format
Download link : 
https://drive.google.com/file/d/1hhcsXxGehgf_wf2QJKSuwB7e3xxrTYn9/view?usp=sharing
![image](https://github.com/user-attachments/assets/65c03d12-3a38-4fe3-b9be-c67fbfa462af)
![image](https://github.com/user-attachments/assets/51a3b91a-eccd-4df3-afbe-d75e17891bf6)  

## Protocol and Constraints
1. We randomly split the AFLW2000 dataset into 10% and 90% as the validation and testing dataset in this challenge
2. Face Detector is not allowed during training
3. The upper bound of model size is 15MB
4. The model should contain weights (such as neural network)
5. We target on float32 solutions. (Float16, int8 or any other quantization methods are not allowed.)
6. You can ONLY train a single model. You cannot train an ensembled model.  

## Methodology or Model Architecture

### Model Architecture

The model used in this experiment is **ShuffleNet V2** [2]. While the original ShuffleNet compresses computation using **group convolutions (GConv)** and **channel shuffle**, ShuffleNet V2 introduces several architectural improvements:

- When the number of input and output channels are equal, **Memory Access Cost (MAC)** is minimized. Thus, after the **Channel Split** in V2, the left branch directly passes through, while the right branch performs convolutions with equal channel sizes.
- Since MAC increases with the number of groups in GConv, ShuffleNet V2 avoids group convolutions in its 1x1 convolutions.
- High branching in networks reduces parallelism, so V2 limits branching via **Channel Split** instead of complex multi-branch structures.
- **Element-wise operations** are computationally expensive and avoided. Therefore, V2 uses **concatenation** for merging branches, including during the stacking of ShuffleNet V2 units.

### Heatmap-based Prediction

We later found literature suggesting the use of **heatmaps** as model output [3]. Inspired by this, we modified the original ShuffleNet V2 model (which directly predicted 68 facial landmarks) by:

- Removing the final fully connected layer.
- Adding two **transposed convolution layers** followed by two **convolution layers**.
- This new architecture outputs a heatmap of size **68 × 96 × 96**, where each channel corresponds to a facial landmark.

## Implementation Details

### Data Preprocessing

Upon examining the datasets, we noticed that the **training images** were generally more uniform in brightness, saturation, resolution, and facial completeness compared to the **validation/testing images**, which exhibited:

- Large variations in brightness,
- Occlusions over facial areas,
- Cropped or partial faces.

To address this, we applied **color jittering** to the training images, randomly altering brightness and saturation, adding occlusions, and converting some images to grayscale. This aimed to make the training data more similar to the testing conditions.

We originally planned to apply rotation-based augmentation as well, but time constraints prevented proper label adjustment, so this step was skipped.

### Hyperparameter Choices

All data augmentations were implemented using **PyTorch built-in functions**. Details include:

- **Color jittering** with brightness, hue, and saturation factors randomly selected between 0.5 and 1.5.
- **Random Erasing** applied with a probability of 0.5.
- **Grayscale conversion** also with a probability of 0.5.

Training parameters:

- **Epochs**: 25  
- **Batch size**: 16  
- **Initial learning rate**: 0.01  
- **Learning rate decay milestones**: at epochs 10, 15, and 20 (decay factor = 0.4)  
- All other parameters were left as model defaults.

### Loss Function (Evaluation Metric)  
Under model constraints, we evaluate the results on the testing dataset with the commonly used Normalized mean error (NME). 
For a matched prediction and the bounding box with width W and height H, we calculate the NME as follows,
![image](https://github.com/user-attachments/assets/49ff1aad-679b-4634-9459-fd68042a9cc0)  
How to calculate the NME by yourself  
![image](https://github.com/user-attachments/assets/097c9411-5b19-45a6-ba63-7928dc374380)  

The project requirement was to use **Normalized Mean Error (NME)** as the loss function to compute the difference between predicted and ground truth coordinates. We implemented a custom loss class for this purpose.

However, during training, we found that the model converged to the same coordinates for different input images. This suggested the custom NME loss function might be non-differentiable or unstable, which prevented gradient updates. Therefore, we reverted to using **PyTorch’s built-in MSE loss**, which provided stable and reliable training.

### Heatmap Conversion

When using heatmaps as model output, the original landmark coordinates (2D) must first be converted to 96×96 heatmaps. This was done via a transformation function that scales the pixel intensities so that:

- Maximum value is 100
- Minimum value is 0

The result is a **localized heatmap** per landmark.  
![image](https://github.com/user-attachments/assets/86b37f98-5c05-4d57-a900-2775e20c06d2)

During inference, these heatmaps are converted back into 2D coordinates using a decoding function that extracts the maximum activation point for each channel.

## Experiment results
The predicted facial landmarks were ploted on the following example image:  
![image00002](https://github.com/user-attachments/assets/c44bc192-10c7-4214-8106-cb51b7d35d56)

| Experiment | Training Setup | Validation Loss |
|-----------|----------------|------------------|
| 1 | 10,000 training samples, no augmentation | 22.78 |
| 2 | Full training dataset, no augmentation | 14.59 |
| 3 | Full training dataset, grayscale only | 5.31 |
| 4 | Full training dataset, color jitter + random erasing + grayscale | 11.604 |

**Observation**: Surprisingly, applying full augmentation (Experiment 4) resulted in worse performance than grayscale-only training (Experiment 3). We suspect that over-augmentation may have introduced too much noise, which degraded the model's learning ability.

## Conclusion (Optional)

While our heatmap-based model did not outperform the direct coordinate prediction version, we believe this is due to insufficient architectural tuning. Adding modules like **attention mechanisms** could improve spatial awareness and performance. Future work should explore these enhancements to fully realize the potential of heatmap-based facial landmark detection.

## References

1. Adrian Bulat, Georgios Tzimiropoulos. “How far are we from solving the 2D & 3D Face Alignment problem? (and a dataset of 230,000 3D facial landmarks)” arXiv:1703.07332v3 [cs.CV].

2. Ma, Ningning; Zhang, Xiangyu; Zheng, Hai-Tao; Sun, Jian. “ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design” arXiv:1807.11164v1 [cs.CV].

3. Wayne Wu, Chen Qian, Shuo Yang, Quan Wang, Yici Cai, Qiang Zhou. “Look at Boundary: A Boundary-Aware Face Alignment Algorithm” arXiv:1805.10483v1 [cs.CV].
