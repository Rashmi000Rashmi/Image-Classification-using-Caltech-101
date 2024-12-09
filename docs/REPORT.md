### Project Title: Image Classification using Caltech101.

**Name of AI system** : Neural Network Classifier

**Type the document**: Project Report

**Course number/name:**  Comp841, Practical AI

**Course instructor:**, Nicholas Noboa.

**Author names**: Shireesha, Shravani and Rashmi

**Date** : 17 Dec 2023


#### 1. AI System Overview
      - Author : Rashmi , Reviewers : Shravani, Shireesha

**Type of AI System:**

Our Team selected to investigate on Neural Network Classifier.
A neural network is a type of Artificial Neural Networks,
which are inspired by the human brain's learning process, comprise neurons that form layers.
These neurons, alternatively referred to as tuned parameters, play a crucial role in the functioning of the neural network. 
The computer can learn and refine its performance through the analysis of new data in a neural network.
This network includes an input layer, one or more hidden layers, and an output layer.
Each node, also known as an artificial neuron, is interconnected with others,
possessing a corresponding weight and threshold.

Our team is chosen 
to work on Image Classification, which is closely related to Neural networks, particularly Convolutional Neural Networks.
CNNs are customized versions of neural networks
that integrate multilayer structures with specific layers
designed to extract crucial and relevant features for the classification of objects.
It have the capability to autonomously identify, generate, and understand features within images.
This significantly lessens the requirement for manual labeling and segmentation of images in preparation for machine learning algorithms.

**AI System:**

We opted for the VGG16 pre-trained model, a specific type of Convolutional Neural Network (CNN).
The architecture of VGG16 is straightforward and symmetric,
comprising convolutional and pooling layers followed by fully connected layers.
VGG16 employs a smaller 3x3 convolutional filter,
reducing the risk of overfitting during training and proving optimal for capturing spatial features in images.
The consistent use of 3x3 convolutions makes the network manageable.
VGG16 has learned to recognize various patterns and objects, providing a solid foundation.

**Sources of information**

We chose this AI system since we are curious about the idea of neural network classifiers and 
have a keen interest in image classification. The next stage of the project is to choose the dataset, which is 
essential to our AI system's that is image classification model. We proceeded to use the Caltec101 dataset, which was
chosen from the insightful website 
“TensorFlow. “Caltech101 | TensorFlow Datasets.” Accessed November 6, 2023. https://www.tensorflow.org/datasets/catalog/caltech101.” 

We are supposed to be familiar with the Caltec-101 dataset. The dataset includes one background clutter class and 
multiple objects that correspond to 101 classes. Each image is labeled with a single object.  
There are around 40 and 800 images in each class, for a total of about 9k images.

**TensorFlow:** TensorFlow, is a popular open-source machine learning framework developed by Google and trustworthy.
- Date: "2015-11-09" , License: Apache-2.0

Abadi, Martín, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, et al. “TensorFlow, Large-Scale Machine Learning on Heterogeneous Systems.” C++, November 2015. https://github.com/tensorflow/tensorflow/tree/v2.15.0

**Repositories:** (Attapanhyar, 2023)
- Author: Attapanhyar , Open-source 

**Example of implementation :** 

**Transfer Learning:** It involves leveraging a pre-trained neural network VGG16  trained on a large dataset (ImageNet).
- Author: Google Inc. , Date: "2023-12-18" , License: Apache-2.0

TensorFlow. “Tf.Keras.Applications.Vgg16.VGG16 | TensorFlow v2.14.0.” Accessed December 18, 2023. https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg16/VGG16. 

**Data Preprocessing:** Preparing and augmenting the dataset for model training and testing.

TensorFlow. “Tf.Keras.Preprocessing.Image.ImageDataGenerator | TensorFlow v2.14.0.” Accessed December 18, 2023. https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator.

**Compilation and Training and Evaluation:** Configuring the model for training it on the prepared dataset.
Assessing the model's performance on the test dataset.

-Author: Jason Brownlee , Date: June 28, 2022, Open source

Brownlee, Jason. “Evaluate the Performance of Deep Learning Models in Keras.”  MachineLearningMastery.Com (blog), June 28, 2022. https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/. 


#### 2. Relevant Theoretical Background
    - Author : Shireesha, Reviewers : Shravani, Rashmi 

Our AI system is designed for image classification using the Caltech-101 dataset. The primary goal is to correctly identify objects within images and assign them to predefined categories.

The image classification AI system uses a pre-trained VGG16 which is a type of convolutional neural network (CNN) as the base model. The system is trained on the Caltech-101 dataset, with data augmentation for the training set and normalization for the testing set. 
The model is compiled with the Adam optimizer and categorical crossentropy loss, and then trained for 20 epochs. The trained model is evaluated on a separate test set, and predictions can be made on new images. 

Need to have a good understanding about the Neural networks. As our AI system uses the VGG16 architecture which is a convolutional neural network(CNN), so we should have the knowledge of both VGG16 and  CNN. 
Training the model involves the dataset splitting, creating the test and train directories, apply data augmentation for images, how the VGG16 architecture is used in building the model, model training and models prediction of image label for input data.

**Neural networks** are composed of layers of interconnected nodes (neurons). Layers include an input layer, hidden layers, and an output layer. Activation functions introduce non-linearity to the model.(IBM,2023)

**Convolutional Neural Networks**, are specialized deep learning architectures designed for image and grid-like data processing. They excel in automatically learning hierarchical features through convolutional layers, which apply filters to capture patterns. 
Additionally, pooling layers reduce spatial dimensions while preserving essential information, making CNNs highly effective for image recognition and classification tasks.(Mishra Mayank. 2020)

**VGG16 Architecture** consists of 16 weight layers. It has multiple convolutional layers with small 3x3 filters and max-pooling layers. 
We should have the knowledge how to apply the VGG16 to our AI system, as it is pretrained on different dataset (medium, 2023 )

**Data augmentation** involves applying various transformations to the existing dataset to artificially increase its size, diversity, and improve the model's generalization. 
Augmentation techniques include rotation, flipping, scaling, translation, shearing, and changes in brightness and contrast.(Tensorflow,2023)

Knowledge on libraries like Tensorflow - used to design the model,
Numpy - used for processing image data stored as arrays and transforming the categorical labels into a binary format, 
Scikit-Learn - used to split the data and to create training and testing datasets and Keras.

#### 3. AI System Development and Evaluation
    - Author : Shravani Reviewers : Shireesha, Rashmi

1. **Data Preprocessing and Splitting:**
   - Upload the zip folder of caltech-101 dataset to the data directory in project directory.
   - unzip caltech-101 using the command - 
         gzip -d this_is_the_file.tar.gz
         tar -xvf this_is_the_file.tar
   - Import the packages tensorflow, keras, numpy, matplotlib, scikit-learn.
   - Store the array of image path in image_data list.
   - Store the labels of all images in labels list. 
   - Split the caltech-101 dataset into train and test sets using `train_test_split` from scikit-learn.       
   - Directories for training and testing data are created (`train_dir` and `test_dir`).
   - Copy the train data to train_dir and test data to test_dir.(caltech101, TensorFlow, 2023)

2. **Data Augmentation and Normalization:**
   - Data augmentation is applied to the training set using `ImageDataGenerator`. This includes rescaling, shearing, zooming, and horizontal flipping.
   - The testing set is normalized (rescaled) but does not undergo augmentation.(TensorFlow, 2023)

3. **Model Architecture:**
   - The VGG16 model is loaded with weights from ImageNet, and its top layer is excluded (`include_top=False`).
   - The layers of the pre-trained VGG16 model are frozen to retain pre-trained features during training.(Makarenko, Andrii. 2023)

4. **Custom Model Head:**
   - A custom model head is added on top of the VGG16 base.
   - A Flatten layer is used to flatten the output from the VGG16 base.
   - A Dense layer with ReLU activation and 256 neurons is added, followed by a Dropout layer for regularization.
   - The final Dense layer with 102 neurons and softmax activation is added for multi-class classification. (V, Nithyashree. 2021)


5. **Model Compilation:**
   - The model is compiled with the Adam optimizer, categorical crossentropy loss, and accuracy as the evaluation metric.

6. **Training:**
   - The model is trained using the `fit` method on the training generator.
   - The training process runs for 30 epochs, and validation data is provided using the testing generator.

7. **Evaluation:**
   - The trained model is evaluated on the testing generator to obtain test loss and accuracy.

8. **Prediction on a Single Image:**
   - A path to a single image is specified, and predictions are made using the trained model.
   - The predicted class label is printed.

9. **Visualization of Training History:**
   - Training and validation accuracy and loss are plotted over epochs using matplotlib. (Figure 1, Figure 2)     

The AI system uses transfer learning, taking advantage of a pre-trained VGG16 model and adjusting it to excel in recognizing images in the Caltech-101 dataset. 
By incorporating data augmentation during training, the model becomes better at understanding a wide range of images. We then check how well it performs on a separate testing set to measure its accuracy and effectiveness.


#### 4. Project Management
    - Authors : Shireesha, Shravani, Rashmi

| Task                                                            | Description                                                              | Status      | Assigned to                 | start date   | End Date     |
|-----------------------------------------------------------------|--------------------------------------------------------------------------|-------------|-----------------------------|--------------|--------------|
| **Initiation:**                                                 |                                                                          |             |                             | Oct 31, 2023 | Nov 06, 2023 |
| AI system and Dataset Selection                                 | Selection of Type of AI System and Dataset                               | Completed   | Rashmi, Shireesha, Shravani |              |              |
| Finalize project proposal                                       | About type of Project, AI system and Tools                               | Completed   | Rashmi, Shireesha, Shravani |              |              |
| **Design:**                                                     |                                                                          |             |                             | Nov 19, 2023 | Nov 20, 2023 |
| Design Document                                                 | Overview of the Project and its Objectives                               | Completed   | Rashmi, Shireesha, Shravani |              |              |
| **Code:**                                                       |                                                                          |             |                             | Nov 21, 2023 | Dec 18, 2023 |
| Load the dataset                                                | Load the Caltech-101 dataset for preprocessing                           | Completed   | Shravani                    |              |              |
| Split the data into train and test                              | Split the data into train and test sets                                  | Completed   | Shravani                    |              |              |
| Design the supervised learning model VGG16                      | Build the VGG16 model Architecture                                       | Completed   | Shireesha                   |              |              |
| Train the neural network to correctly classify the test dataset | Train the model to predict and classify the images                       | Completed   | Shireesha                   |              |              |
| **Testing:**                                                    |                                                                          |             |                             | Dec 06, 2023 | Dec 18, 2023 |
| Overfit the neural network                                      | Examine the model's performance on both the training and validation sets | Completed   | Rashmi, Shireesha, Shravani |              |              |
| Input the test image to test the model                          | Give the input text image for model to predict                           | Completed   | Rashmi                      |              |              |
| Graph the model performance                                     | Models performance graph                                                 | Completed   | Rashmi                      |              |              |               
| **Documentation:**                                              |                                                                          |             |                             | Dec 11, 2023 | Dec 18, 2023 |
| Finalize README.md                                              | Description of project type and the Model                                | Completed   | Shravani, Shireesha, Rashmi |              |              |
| Finalize HOWTO.md                                               | start to end procedure of project                                        | Completed   | Shireesha, Shravani, Rashmi |              |              |
| Project final Report                                            | Documentation of Model's performance                                     | In progress | Shireesha, Shravani, Rashmi |              |              |


#### 5. References

“Boost Your Image Classification Model with Pretrained VGG-16 | by Andrii Makarenko | Geek Culture | Medium.” n.d. Accessed December 18, 2023. https://medium.com/geekculture/boost-your-image-classification-model-with-pretrained-vgg-16-ec185f763104.

“Tf.Keras.Preprocessing.Image.ImageDataGenerator | TensorFlow v2.14.0.” n.d. TensorFlow. Accessed December 18, 2023. https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator.

“What Are Neural Networks? | IBM.” n.d. Accessed December 18, 2023. https://www.ibm.com/topics/neural-networks

Mishra, Mayank. 2020. “Convolutional Neural Networks, Explained.” Medium. September 2, 2020. https://towardsdatascience.com/convolutional-neural-networks-explained-9cc5188c4939.

“Caltech101 | TensorFlow Datasets.” n.d. TensorFlow. Accessed December 18, 2023. https://www.tensorflow.org/datasets/catalog/caltech101.

Makarenko, Andrii. 2023. “Boost Your Image Classification Model with Pretrained VGG-16.” Geek Culture (blog). March 26, 2023. https://medium.com/geekculture/boost-your-image-classification-model-with-pretrained-vgg-16-ec185f763104.

V, Nithyashree. 2021. “Step-by-Step Guide for Image Classification on Custom Datasets.” Analytics Vidhya (blog). July 19, 2021. https://www.analyticsvidhya.com/blog/2021/07/step-by-step-guide-for-image-classification-on-custom-datasets/.

“DeepLearning/101caltec_with_VGG16[SOLVED].Pdf at Main · Attapanhyar/DeepLearning.” n.d. GitHub. Accessed December 18, 2023. https://github.com/attapanhyar/DeepLearning/blob/main/101caltec_with_VGG16%5BSOLVED%5D.pdf.

“How Does Image Classification Work? - Unite.AI.” n.d. Accessed December 18, 2023. https://www.unite.ai/how-does-image-classification-work/.

“Classification Model Using Artificial Neural Networks (ANN).” n.d. upGrad Blog. Accessed December 18, 2023. https://www.upgrad.com/blog/classification-model-using-artificial-neural-networks/.

Boesch, Gaudenz. 2021. “VGG Very Deep Convolutional Networks (VGGNet) - What You Need to Know.” Viso.Ai. October 6, 2021. https://viso.ai/deep-learning/vgg-very-deep-convolutional-networks/.


#### 6. Appendix

![Train_and_validation_accuracy.jpg](..%2Fdata%2FTrain_and_validation_accuracy.jpg)

Figure 1

![Train_and_validation_loss.jpg](..%2Fdata%2FTrain_and_validation_loss.jpg)

Figure 2

