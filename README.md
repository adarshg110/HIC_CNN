# HIC_CNN
In Remote Sensing, Hyperspectral remote sensors are widely used for monitoring the earth’s surface with a high spectral resolution. Hyperspectral Image(HSI) data often contains hun- dreds of spectral bands over the same spatial area which provide valuable information to iden- tify the various materials. In HSI, each pixel can be regarded as a high dimensional vector whose entries correspond to the spectral reflectance from visible to infrared. The Classification of Hyperspectral Images is the task of classifying a class label of every pixel in an image that was captured using the hyperspectral sensors. In this Project, we have surveyed the traditional learning method over the HSI and then further explored the Deep learning models in detail to classify Hyperspectral images. The models proposed are 2Dimensional and 3 Dimensionsal Convolutional Neural Network and Diverse based region CNN .

Hyperspectral image classification is a process that involves assigning land cover categories or classes to pixels in a hyperspectral image. A hyperspectral image is a multi-dimensional image that contains hundreds or thousands of contiguous narrow wavelength bands, which makes it possible to identify the materials present in an image with a high degree of accuracy.

The process of hyperspectral image classification typically involves the following steps:
1. _Data preprocessing_: The first step in hyperspectral image classification is to preprocess the data. This may include correcting atmospheric effects, removing noise, and filling in missing data.
   
2. _Feature extraction_: The next step is to extract features from the hyperspectral image. This may involve applying algorithms such as principal component analysis (PCA) or independent component analysis (ICA) to reduce the dimensionality of the data.

3. _Classification_: The final step is to classify the pixels in the image into different land cover classes. This can be done using a variety of algorithms, such as decision trees, support vector machines (SVMs), and neural networks.

Hyperspectral image classification is commonly used for tasks such as land cover mapping, mineral exploration, and environmental monitoring. It is a specialized type of image classifi- cation that requires specialized algorithms and techniques due to the high dimensionality and complexity of the data.

**2D CNN approach**
This method inspired is based on a coherent structure that incorporates spectral and spatial information in a single stage, creating high-level spectral-spatial characteristics at the same time. Specifically,Modified Convolutional Neural Network (CNN) is used that performs the operation of constructing large-level features and a MultiLayer Perceptron (MLP) which is used for the classification of the image. Due to the presence of the feed forward network in CNNs and MLPs, the improved framework simultaneously constructs spectral-spatial characteristics under this type of architecture and performs real-time predictions of the various classes in the image.
The spectral values of pixels with similar class labels for a channel are similar, whereas those with differing class labels are distinct. Based on these characteristics, a _dimensionality reduction_ approach may be used to reduce the dimensionality of the input data and improve the training and classification processes. The dimensionality reduction approach known as _Principal Component Analysis(PCA)_ decreases the spectral dimensions of the hyperspectral image without sacrificing any image data. Split the hyperspectral image into small patches after dimensionality reduction to make it compatible with CNN’s fundamental structure. The spectral and spatial characteristics of a single pixel are contained in each produced patch.

**CNN architecutre**





_**Training Specifications:**_
• Patch size = 13*13
• Test size = 0.2
• Adam Optimizer with learning rate of 0.001
• Size=256
• Epochs = 80 with early stopping of patience 5.

_**Results from our implementation:**_
• Training Accuracy = 99.6
• Test Accuracy = 98.44
• Average Accuracy = 96.76 
• Kappa Coefficient = 0.9948

**3D CNN approach**

As seen before, both the spectral factor and the spatial factor influence the class label prediction of a pixel. On one hand, the label of a pixel is reflected by its spectral values scanned by using different spectra. On the other hand, as the geographically close pixels tend to belong to the same class, predicting the class label of a pixel should take into account the class labels of the surrounding pixels. Hence, a good hyperspectral image classification method should consider both the spectral factor and the spatial factor.
The spatial context can be used by the 2-D-CNN model, but the spectral correlations are not taken into account. We develop a 3-DCNN model with seven convolutional layers and one complete connection layer to overcome this issue. This model’s convolution operator is 3- D as opposed to 2-D-CNN, with the first two dimensions being used to capture the spatial context and the third dimension to capture the spectral context. Despite having more network parameters than its 2-D equivalent, the 3-D-CNN model should be more efficient since it can assess the spectral correlations of a hyperspectral image.
One main difference between a hyperspectral image and a conventional image is that the for- mer is captured by scanning the same region with different spectral bands, while the latter is not. As the image formed by hyperspectral bands may have some correlations, e.g., close hyperspectral bands may result in similar images, it is desirable to take into account hyper- spectral correlations. Though the 2-D-CNN model can utilize the spatial context, it ignores the hyperspectral correlations. Hence, we develop a 3-D-CNN model

_** Illustration of 3D convolution**_
 
The primary distinction is that the 3-D-CNN model has an additional reordering phase. The hyperspectral bands are rearranged in this phase in ascending order. This sequential ordering of images of related spectral bands can maintain the correlations between them in a spectral context. The two models’ patch extraction and label identification phases are remarkably comparable. Instead of using a 2-D convolution operator for the feature extraction stage, the 3-D-CNN model is used.

_**Model architecutre:**_



**_Training specifications_**:

• Patch size = 7*7*200
• Test size = 0.2
• Adam Optimizer with learning rate of 0.001
• Size=32
• Epochs = 80 with early stopping of patience 5.

**_Results from our implementation:_**
• Training Accuracy = 98.75 
• Test Accuracy = 97.07
• Average Accuracy = 98.17
• Kappa Coefficient = 0.9858

**Model accuracy plot**









