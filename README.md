"# Facemask-type-AI Project"

Team members
1. Kary Sutariya (40193909)
2. Maryam Valipour (40224474) 
3. Kanika Aggarwal (40195981)

To run this code, all required libraries which has been used in training, evaluation and application python files must be installed. 
training.py: By running this file, it will generate the trained models.
evaluation.py: By running this file, the generated models from training.py will be evaluated on the test dataset.
application.py: By running this file, real-time data from SampleData Folder will be passed to the model to predicted their labels.

List of file:
1. Data folder -  this folder contain images of five categories "Cloth mask", "No face mask",
"Surgical mask", "N95 mask", "Mask worn incorrectly". 
2. training.py - code for face mask detection with data loading, data cleaning and preprocessing, with model traing. Training and testing data split has been done using Sklearn with random_state so again same splited data can be used in evalution phase.   
4. evaluation.py - it uses models generated from model training for evaluation.
5. application.py - this file is for real time application. it reads images from SampleData Folder which have not been used in either training or testing phase.  
6. Figures - this folder contains the figures used in project report.
7. SampleData - few images from all the five classes in this folder for real-time application purpose.
8. Models - this folder stores both models and the model's loss and accuracy in training and validaiton. To test the code, please download model from the given google drive link and put it in corresponding folders (Models/ver1, Models/ver2, and Models/ver3).

Models/ver1 - https://drive.google.com/file/d/1qwdwPGtYSvfjoYVto7LWoHeJ3DN_GlgY/view?usp=sharing
Models/ver2 - https://drive.google.com/file/d/1-lxymdGwY50GvSck-A2WFhpD5lSwSM8Q/view?usp=sharing
Models/ver3 - https://drive.google.com/file/d/1ebwzJUXSjOn7HY8MIWp2i56SdZAFvTfG/view?usp=sharing


