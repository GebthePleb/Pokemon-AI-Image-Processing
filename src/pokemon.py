import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten

# defining root directory
from PIL import Image



# root_dir = r"assets\images"
# root_dir = r"assets\images_b"
# root_dir = r"assets\images_flipped"
root_dir = r"C:\Users\1opal\Documents\GitHub\Pokemone-AI-Image-Processing\assets\images"
# root_dir = r"C:\Users\1opal\Documents\GitHub\Pokemone-AI-Image-Processing\assets\images_b"
# root_dir = r"C:\Users\1opal\Documents\GitHub\Pokemone-AI-Image-Processing\assets\images_flipped"


files =  os.path.join(root_dir)
File_names = os.listdir(files)

#Used to see that all the images are in the directory
#print("This is the list of all the files present in the path given to us:\n")
#print(File_names)

# plot here
# fig, axes = plt.subplots(2, 3, figsize=(15, 8))
# first_five = File_names[0:6]

# def subplots():
# # Use the axes for plotting
#     i = 0
#     j = 0
#     k = 0
#     for k in range(5):
#         state = os.path.join(root_dir, first_five[k])
#         img = Image.open(state)
#         axes[i,j].imshow(img)
        
#         if k==2:
#             i +=1
#             j = 0
#         else:
#             j += 1


#     plt.tight_layout(pad=2)
#     #prints the first 5 pokemone images
#     #plt.show()
    
# subplots()



# data = pd.read_csv(r"assets\pokemon_labels.csv")
data = pd.read_csv(r"C:\Users\1opal\Documents\GitHub\Pokemone-AI-Image-Processing\assets\pokemon_labels.csv")



#use this to see that the csv file is working and printing
#print(data.head())


#now we want to extrat the name and type1 row for our purposes
data_dict = {}
for key, val in zip(data["Name"], data["Type1"]):
    data_dict[key] = val

#this will print out the name : type1 for each item in the dict
#print(data_dict)


#Now we need to extract the pokemone type in the list that are possible answers
labels = data["Type1"].unique()
#to see all 18 are there
#print(labels)

#this will turn the string verions of the types into a corrosponding number
ids = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
number_labels = dict(zip(labels,ids))

print(number_labels)


#this was for testing to see if the image is working
# img = cv2.imread(os.path.join(root_dir, File_names[]), cv2.COLOR_BGR2GRAY) 
# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


final_images = []
final_labels = []
count = 0
files =  os.path.join(root_dir)

for file in File_names:
    count += 1
    img = cv2.imread(os.path.join(root_dir, file), cv2.COLOR_BGR2GRAY) 
    label = number_labels[data_dict[file.split(".")[0]]] 
    # append img in final_images list
    final_images.append(np.array(img))
    # append label in final_labels list
    final_labels.append(np.array(label))

#Testing to make sure one is correct. In this case growlithe
# cv2.imshow('img',final_images[18])
# print(final_labels[18])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

final_images = np.array(final_images, dtype = np.float32)/255.0
final_labels = np.array(final_labels, dtype = np.int8).reshape(809, 1)


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(120, 120,3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(18, activation='softmax')
])

model.compile(optimizer='rmsprop',
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

print("Model Creation")

history = model.fit(final_images, final_labels, epochs=10, batch_size=32,
                    validation_split=0.1)

print("Model Done")




val_ac = history.history["val_accuracy"]
epochs = range(1, 11)
plt.plot(epochs, val_ac, "b--",
         label="Validation accuracy")
plt.title("Validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
