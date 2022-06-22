import os

from PIL import Image
from sklearn.model_selection import train_test_split


DIRECTORY = '../Data/Data part 2/'
CATEGORIES = ['Cloth mask','Mask worn incorrectly','N-95_Mask','No Face Mask','Surgical Mask']

data = []
name = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)

    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        
        try:
            image = Image.open(img_path)
            data.append(image)
            name.append(img)
            labels.append(category)
        except:
            pass  

total_images = len(data)
train_dataset_size = 1570
test_dataset_size = total_images - train_dataset_size
test_dataset_percentage = test_dataset_size/total_images
data_train, data_test, labels_train, labels_test, name_train, name_test = train_test_split(data, labels, name, test_size=test_dataset_percentage, random_state=42,stratify = labels)


# f = open('make_annotation.csv','w')
# for im,la,na in zip(data_test,labels_test, name_test):
#     line1 = "../Data part 2/"+la+"/"+na+"\n"
#     f.write(line1)
# f.close()



# with open('make_annotation.csv', 'r') as file:
#     reader = csv.reader(file)
#     for row in reader:
#         print(row[0])

test_path = DIRECTORY+'Test/'
isExist = os.path.exists(test_path)

if not isExist:
    
    os.makedirs(test_path)
    os.mkdir("../Data/Data part 2/Test/N-95_Mask/")
    os.mkdir("../Data/Data part 2/Test/Mask worn incorrectly/")
    os.mkdir("../Data/Data part 2/Test/No Face Mask/")
    os.mkdir("../Data/Data part 2/Test/Surgical Mask/")
    os.mkdir("../Data/Data part 2/Test/Cloth mask/")


for im,la,na in zip(data_test,labels_test, name_test):
    im.save("../Data/Data part 2/Test/"+la+"/"+na)
 