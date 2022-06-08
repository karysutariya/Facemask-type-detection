"# Facemask-type-AI Project"

Team members
1. Kary Sutariya (40193909)
2. Maryam Valipour (40224474) 
3. Kanika Aggarwal (40195981)

List of file:
1. Data folder -  this folder content images of five categories "Cloth mask", "No face mask",
"Surgical mask", "N95 mask", "Mask worn incorrectly". 
2. training.py - code for face mask detection with Data loading, data cleaning and preprocessing, with model traing. Training and testing data split has been done using Sklearn with random_state so again same splited data can be gained in evalution part for testing.   
4. evaluation.py - it test models which has been created after model training.
5. Application.py - this file is for real time application. it reads images from Sample Data which haven't used in either training or testing phase.  
6. Figures - this folder contains images which are generated for project report
7. Sample Data - few images are there in this folder for evaluation purpose.
8. Models - this folder stores models. To test the code, please download model from given link of google drive and put it in Models/Ver1 folder

Models/ver1 - https://drive.google.com/file/d/1qwdwPGtYSvfjoYVto7LWoHeJ3DN_GlgY/view?usp=sharing
Models/ver2 - https://drive.google.com/file/d/1-lxymdGwY50GvSck-A2WFhpD5lSwSM8Q/view?usp=sharing
Models/ver3 - https://drive.google.com/file/d/1ebwzJUXSjOn7HY8MIWp2i56SdZAFvTfG/view?usp=sharing

To run this code, all required libraries which has been used in training, load_data, evaluation and application python files must be installed. Then one can simply execute all code blocks subsequently. This jupyter file includes both training and application part. During evalution part, please avoid traing blocks


Note: In this code, we have implemented three different models. Here, one should take care of "load_model = False" so if this condition is false for a model, it won't be loaded. By default, this condition is true for main model (model no 1)
