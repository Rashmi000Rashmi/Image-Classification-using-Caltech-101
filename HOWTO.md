## Image classification
- **Contributors** : Shireesha, Shravani, Rashmi
- **Date** : 12/16/2023

### Dataset Selection:
- Selection of Clean and classic dataset from Tensorflow 
- Dataset Name: Caltech-101
    - TensorFlow dataset : https://www.tensorflow.org/datasets  

- Download the Dataset Caltech-101 through https://data.caltech.edu/records/mzrjq-6wc02

### Platforms
- Sagemaker Studio Labs : Ensure that you have registered and signed in to sagemaker studio labs. 
You can register in https://studiolab.sagemaker.aws/

- Git : Make sure you have registered for GitHub and signed in to GitHub. 
You can register in https://github.com/

### Cloning the Repository
1. Open *comp841* folder in PyCharm
2. Open a Terminal (select git-bash, if you have Windows), cd into 
   project. 
3. Copy the repository URL from the project's GitHub or other version control platform.
4. In the terminal or command prompt, run the following command, replacing <repository_url> with the actual URL
5. git clone <repository_url>

Example : git clone https://github.com/your-username/your-repository.git

### Setup virtual environment

- Check the python version using command `python --version`.
- If python version is not 3.11 add python version as 3.11 while creating the environment.
- Create a virtual environment by using the command `conda create -n <env name> python=3.11
- Activate environment by using the command `conda activate <env name>
- To display the only lines containing ipykernel installed in python environment run the command `pip freeze | grep ipykernel` 
- Install `ipykernel` with `conda install ipykernel`. This lets the virtual environment be used with JupyterLab.
- Open the `image_classifier.ipynb` jupyter notebook present in src folder and select the kernel as `classifier` in the top right corner.

This environment is used with the `image_classifier.ipynb` notebook in `project`

### Dependencies
- In the terminal Ensure to activate the conda environment and Navigate to project directory.
- Install the following packages
    - Tensorflow 
    - Keras
    - Scikit-learn
    - matplotlib
    - numpy

- This can be done with one command: `pip install keras tensorflow scikit-learn pillow matplotlib`
- Verify the installation: `pip list`
- Create the `environment.yml` in project repository using command `touch environment.yml`
- To store the installed packages in environment.yml file run the command `pip freeze > environment.yml`

### Implementation

- Open the Jupyter Notebook `image_classifier.ipynb` in src folder.
- Upload the Caltech-101 dataset in data folder
- Unzip the Caltech-101 folder using command in terminal `unzip <file name>`
- In Caltech-101 there are 2 folders 101_ObjectsCategories.tar.gz and Annotations.tar. untar the folders using commands 
`gzip -d this_is_the_file.tar.gz`
`tar -xvf this_is_the_file.tar`
- Import the packages Tensorflow ,Keras , Scikit-learn , matplotlib, numpy.
- Import ImageDataGenerator, VGG16 and image from Tensorflow  
- Create the objects folder inside Caltech-101 
- Split the data into train and test set and store it in objects folder.
- Build the model using VGG16 architecture.
- Train the model to predict and  calssify the test Images.
- Examine the model's performance on both the training and validation sets.
- Plot the models accuracy and loss  in a graph using matplotlib.
- Give an input image to the model to predict and classify the label. 
- Navigate to kernal and select restart the kernal and run all cells.
- or Run the cells individually by selecting the cell and do shift+enter.

