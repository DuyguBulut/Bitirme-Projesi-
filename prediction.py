
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import operator
import random
import glob
import os.path
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

BATCH_SIZE = 120
SEED_VAL = 100

MODEL = load_model('C:\\Users\\duygu\\Desktop\\design.005-0.00.hdf5')
test_dir  = "C:\\Users\duygu\\Desktop\\testSet"
alphabet_classes = ['A','B','C','Ç','D','E','F','G','Ğ','H','I','İ','J','K','L',
           'M','N','O','Ö','P','R','S', 'Ş' ,'T','U','Ü','V', 'Y','Z']

def get_test_generator():
    
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        classes =alphabet_classes,
        class_mode='categorical',
        seed = SEED_VAL)

    return test_generator


test_generator = get_test_generator()

nb_batch = len(test_generator)


#predicted = MODEL.predict_generator(test_generator, verbose=1)
#predicted = np.argmax(predicted, axis = 1)

gtruth = []
gtruth = np.array(gtruth)

predicted = []
predicted = np.array(predicted)

for i in range(nb_batch):
    
    if i == 0:
        gtruth = np.argmax(test_generator[i][1], axis=1)
        
        predicted = MODEL.predict(test_generator[i][0])
        predicted = np.argmax(predicted, axis=1)
        
    else:
        gtruth = np.concatenate((gtruth, \
                np.argmax(test_generator[i][1], axis=1)))

        predicted = np.concatenate((predicted, \
                np.argmax(MODEL.predict(test_generator[i][0]), axis=1)))
        
print("pred_sh:", predicted.shape)
print("gtruth_sh:", gtruth.shape)


correct = 0
classes = alphabet_classes
for i in range(predicted.shape[0]):
    
    pred_label = classes[predicted[i]]
    gtruth_label = classes[gtruth[i]]
    
    if pred_label == gtruth_label:
        correct += 1
        #print(pred_label,gtruth_label)
    
acc = correct / predicted.shape[0] * 100

print("accuracy:", acc)

count = 0
correct = 0
all_classes = alphabet_classes
classes = all_classes[1:-2] + [all_classes[-1]]

for i in range(predicted.shape[0]):
    
    pred_label = all_classes[predicted[i]]
    gtruth_label = all_classes[gtruth[i]]
    
    if gtruth_label in classes:
    
        if pred_label == gtruth_label:
            correct += 1
            #print(pred_label,gtruth_label)
            
        count += 1
        
acc = correct / count * 100

print("accuracy:", acc)

#print conf matrix

def plot_confusion_matrix(y_true, y_pred,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    classes = alphabet_classes
    np.set_printoptions(precision=2)#digit floats 
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        fig_name = "confusion_matrix_nor.png"
    else:
        print('Confusion matrix, without normalization')
        fig_name = "confusion_matrix.png"

    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(10)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(fig_name)
    return ax

plot_confusion_matrix(gtruth, predicted,
                          normalize=True)
 
plot_confusion_matrix(gtruth, predicted,
                          normalize=False)                         
