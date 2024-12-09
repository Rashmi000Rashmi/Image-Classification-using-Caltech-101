### Project Title: Image Classification using Caltech101.

**Authors** : Shravani, Shireesha and Rashmi.

**Date** : 06-11-2023

#### 1. What type of AI system will the team investigate?
    -- Authors: Shireesha, Shravani and Rashmi
We are training a neural network classifier for the `Caltech101` dataset using a `VGG16-based model` with transfer learning. 
The type of AI system that our team is investigating is VGG16-based model which is a type of `CNN`. This system is designed to recognize and classify objects within images into one of the 
101 predefined categories present in the Caltech101 dataset. Our goal is to develop a model capable of accurately identifying objects in images and categorizing them.

#### 2. What AI system has the team selected from publicly available sources to exemplify the type of system under investigation?
    -- Authors: Shireesha, Shravani and Rashmi
Our team selected the `caltech101` dataset which uses VGG16-based model is a type of convolutional neural network.

As Caltech101 is a smaller dataset using VGG16 that has been trained on a large dataset which consists of various objects, animals, and scenes, helps our model 
speed up training and improving classification accuracy.
Instead of training the model from scratch which may lead to overfitting, starting with a pre-trained VGG16 model, the code takes advantage of the 
general image recognition capabilities the model has learned during its extensive training. This often results in better accuracy on the target task (Caltech-101 image classification).
The model is based on the VGG16 architecture, which is composed of a series of convolutional and pooling layers followed by fully connected layers. VGG16 is known for its simplicity and effectiveness in feature extraction from images.
The 16 in VGG16 refers to 16 layers that have weights. There are 13 convolutional layers in VGG16, 5 Max pooling layers, and three Dense layers but it has only 12 weight layers i.e., learnable parameters layer. 
VGG16 takes input tensor size as 224,224 with 3 RGB channel. Therefore, This approach enhances our model's performance on the specific task of Caltech-101 image classification.

#### 3. What content knowledge (theories and concepts) are needed to understand what the AI system does and how it works?
    -- Author: Shireesha
The knowledge needed to understand the AI system and how it works are:

Familiarity with the `Caltech-101 dataset`, which is a collection of `9146 images` belonging to `101 different object categories` with `200 * 300 pixels`. 

Understanding the concept of `Supervised learning` where the models are trained on labeled data.

Should have knowledge of concepts like `deep neural networks` and `Convolutional neural networks`.

**Deep Neural Networks:** It is an artificial neural network with multiple layers between the input and output layers. It is helpful to classify the task based on the features.

**Convolutional neural networks:** It is an artificial neural network which is designed for analyzing visual data like images and videos.  

Pre-trained CNN models like `VGG16` are useful for `feature extraction` and `initial image processing`.

Knowledge on libraries like `TensorFlow`, `Keras` which are useful to train and build neural networks.

Concepts related to splitting data into training and testing sets for `model evaluation`.

Knowledge on `Data Augmentation` and how they can improve model generalization.

**Data Augmentation:** It is a technique of artificially increasing the training set by creating modified copies of a dataset using existing data.

Understanding the concept of `loss functions` used to measure the model's error during training.

Understanding the training process, including `epochs` and `batch size`.

**Epochs:** One complete pass through the dataset

**Batch size:** number of samples processed in each iteration.

Familiarity with metrics like `accuracy`, `precision`, `recall` and `F1 score` for assessing model performance.

#### 4. What tools, platforms, APIs, libraries, and data sets are needed to develop the AI system?
    -- Author: Shravani
Following are the tools, platforms, libraries, and dataset are needed to develop our AI system.

**Tools**: `ImageDataGenerator(from keras)`-It is like a tool which helps us to create more examples of images to train the model. It takes existing images and applies the different types of changes to them like rotating, flipping, shifting, zoom in or zoom out which helps the model to classify the images more accurately by learning the variety of examples.   

**Platforms**:  Need PyCharm for creating the project and running the python code.

**Libraries**: `TensorFlow` - TensorFlow is used to design the model, set the learning rules like optimizer and loss function. It trains the model on data and evaluates its performance by measuring accuracy. `NumPy` - It is used for processing image data stored as arrays and transforming the categorical labels into a binary format that is suitable for training a neural network. `Scikit-Learn` - ‘train_test_split’ function from sklearn is used to split the data and to create training and testing datasets for model development and evaluation. `Keras` - It is used to define the neural network architecture, compile the model by specifying how it should be trained, and perform data augmentation for improving model generalization. Using keras, we can create and describe how our neural network should look and it helps us make our model smarter by showing it more varied examples.

**Dataset**: `caltech101 dataset` - It consists of pictures of objects belonging to 101 classes like “helicopter”, “elephant”, and “chair”, etc, and background categories that contain the images not from the 101 object categories. Each class contains roughly 40 to 800 images, totaling around 9k images.

#### 5. What are the ethical considerations for:
   
    #### a. determining the legitimate use of the AI system and for
    -- Author: Rashmi
**Goal and Setting**:The AI system's goal is to use the Cal-tech101 dataset to build and train a 
convolutional neural network (CNN), a machine learning model, 
for the purpose of classifying images. 
Our project uses an AI system for tasks involving image identification and categorization.

**Ethical Consideration**: "In this project, the dataset and model will be sourced from a real-world application, 
and we will ensure that they will not be used to produce unfair or biased results. 
This will be especially important when it comes to classifying sensitive content or individuals."

**Adherence to the Law**: The VGG16 model, which comes from approved sources, and
The cal-tech101 dataset is used in our project.
It is verified that all copyright and data licensing requirements are met.

    #### b. developing a safe and trustworthy AI system?
    -- Author: Rashmi
**Safe AI:**

**Ethical Property—Reflecting Human Values:** AI systems that are safe are made to respect and replicate human values.
This indicates that they have been socialized and trained to make choices and behave in ways that are ethical 
and in line with accepted social norms.
Fairness, non-discrimination, and respect for the rights and 
beliefs of people as well as communities are given top priority under these systems.

**Explain-ability Property - Transparent Decision-Making:** Transparency is another essential component of safe AI.
These systems are designed to be able to explain the justifications and reasoning that go into their decisions. 
By ensuring
that consumers and other interested parties are able to understand the AI's reasoning behind any given result,
this transparency promotes accountability and confidence.

**Safety Property - Harm-Free Inference and Learning:** The foundation of safe AI is the safety feature. 
It implies that artificial intelligence (AI) systems are designed to be able to draw conclusions, 
adjust, and learn from their surroundings without harming people or the real world. 
This involves avoiding behaviors that could have negative consequences,
limiting errors, and being flexible to unexpected situations.

**Trustworthy AI system**
Properties of Trustworthy Artificial Intelligence (AI) Systems

**Accuracy:** Fundamental to trustworthy ML and AI,
ensuring reliable performance in making correct decisions, even with new data.
To ensure accuracy, ML techniques make use of statistical models and inference algorithms.
It involves replicating training data on fresh data with consistent performance.
It is important to find a balance between accuracy and other important attributes.

**Robustness:** Robustness and accuracy are closely related concepts. 
Even when inputs are purposely indifferent or drastically differ from the training set, a reliable 
AI systems can still identify them with accuracy.
It guarantees the system's ability to manage a variety of input conditions.

**Accountability:** Since accountability deals with taking ownership of the results generated by the artificial
intelligence system, it is essentially related to safety.
While accountability covers who or what organization is responsible for paying for the system's activities, safety 
focuses on preventing harm.
It also admits that outcomes may not be accurate or robust.
Knowing who is responsible for what in the system is crucial, particularly when problems occur.

**Transparency:** One feature that is especially important to artificial intelligence (AI) systems is transparency. 
It has to do with how easily other observers, such as end users, can understand the process by means of 
which an AI system produces its results.

**Explain-ability:** Explain-ability and clarity improve usability by making it simpler for people to 
understand the reasons behind the AI system's capability to generate a certain result.
It is essential that the system's results be supported by explanations that end users can understand or find meaningful.

**Ethical:** The ethical property is related to transparency and includes privacy concerns. 
The gathering, processing, and use of data must adhere to ethical standards in order to promote 
confidence and guarantee a reliable AI system.


#### 6. References: What trustworthy and publicly available materials and resources inform and guide the project work of the team?
    -- Authors: Shireesha, Shravani and Rashmi
TensorFlow. “Caltech101 | TensorFlow Datasets.” Accessed November 6, 2023. https://www.tensorflow.org/datasets/catalog/caltech101.

Learning, Great. “Everything You Need to Know about VGG16.” Medium (blog), September 23, 2021. https://medium.com/@mygreatlearning/everything-you-need-to-know-about-vgg16-7315defb5918.