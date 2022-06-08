"# Facemask-type-AI Project"

Team members
1. Kary Sutariya (40193909)
2. Maryam Valipour (40224474) 
3. Kanika Aggarwal (40195981)

List of file:
1. Data folder -  this folder content images of five categories "Cloth mask", "No face mask",
"Surgical mask", "N95 mask", "Mask worn incorrectly". 
2. main.ipynb - code for face mask detection with Data loading, data cleaning and preprocessing, model traing with testing and evaluation. 
3. Figures - this folder contains images which are generated for project report
4. Sample Data - few images are there in this folder for evaluation purpose.
5. Models - this folder stores models. To test the code, please download model from given link of google drive and put it in Models/Ver1 folder

Models/ver1 - https://drive.google.com/file/d/1mSENhY2ni9cYFSFSBjk1grNlCH-mtt_j/view
Models/ver2 - https://drive.google.com/file/d/1_0pLvvQie4WYhIyy6TjrhTLN5XUUJJcL/view
Models/ver3 -

To run this code, all required libraries which has been used in main.ipynb must be installed. Then one can simply execute all code blocks subsequently. This jupyter file includes both training and application part. During evalution part, please avoid traing blocks


Note: In this code, we have implemented three different models. Here, one should take care of "load_model = False" so if this condition is false for a model, it won't be loaded. By default, this condition is true for main model (model no 1)