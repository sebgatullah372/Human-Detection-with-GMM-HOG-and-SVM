from sklearn.svm import LinearSVC
#from sklearn.externals import joblib
import glob
import os
from skimage.feature import hog
import numpy as np
import cv2
from sklearn.externals import joblib
import pandas as pd
import random
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score , recall_score , classification_report, confusion_matrix

def shuffle_in_place(array):
    array_len = len(array)
    assert array_len > 2, 'Array is too short to shuffle!'
    for index in range(array_len):
        swap = random.randrange(array_len - 1)
        swap += swap >= index
        array[index], array[swap] = array[swap], array[index]
def shuffle(array):
    copy = list(array)
    shuffle_in_place(copy)
    return copy



MODEL_PATH = 'models'
RANDOM_STATE= 31

#pos_im_path = 'Hum\\pos'
#neg_im_path = 'Hum\\neg'
pos_feat= 'features\\pos'
neg_feat= 'features\\neg'
labels = []    
samp=[]
# Get positive samples
for feat_path in glob.glob(os.path.join(pos_feat, '*.feat')):
    #image = cv2.imread(filename, 0)
    #hist =  hog(image, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(2, 2), block_norm='L2', visualise=False, transform_sqrt=False, feature_vector=True)
    x = joblib.load(feat_path)
    labels.append(1)
    samp.append(np.array(x[0:6480]))

# Get negative samples
for feat_path in glob.glob(os.path.join(neg_feat, '*.feat')):
    #img = cv2.imread(filename, 0)
    #hist =  hog(img, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(2, 2), block_norm='L2', visualise=False, transform_sqrt=False, feature_vector=True)
    x = joblib.load(feat_path)
    #print(x)
    labels.append(0)
    samp.append(np.array(x[0:6480]))

# Shuffle Samples
#samp = shuffle(samp)
#labels= shuffle(labels)
#c= list(zip(samp,labels))
#random.shuffle(c)
#samp,labels= zip(*c)
sample= list(samp)
#label= list(labels) 

sample_df = pd.DataFrame()
sample_df["Sample"]=pd.Series(sample)
sample_df["Label"] = pd.Series(labels)

sample_df.head()

sample_df.sample(frac=1)


sample_data = sample_df["Sample"]
sample_label = sample_df["Label"]

#samp= samp.reshape(1,-1)
#labels= labels.reshape(1,-1)
sample_data_reshaped=[]
print(len(sample_data))
for i in sample_data:
    i.reshape(1,6480)
    sample_data_reshaped.append(i)
    #print(i)
sample_data_reshaped = np.array(sample_data_reshaped)
sample_data_reshaped.reshape(len(sample_data_reshaped),6480,1)
X_train, X_test, y_train, y_test = train_test_split(sample_data_reshaped, sample_label, test_size=0.50, random_state=42)    
#x_train = sample[0:6480]
#x_train= samp[2200:3634]    
#y_train=labels[0:6480]
#y_train= labels[2200:3634]
#x_test=samp[5200:6480]
#y_test=labels[5200:6480]
clf = LinearSVC(random_state=RANDOM_STATE)
clf.fit(X_train,y_train)
eval_model=clf.score(X_train, y_train)
#predictions = classifier.predict_classes(sample_data_reshaped)
predictions = clf.predict(X_test)
print(classification_report(y_test,predictions))
print("Model Accuracy ",eval_model*100)
tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
print(tn, fp, fn, tp)
#pred = clf.predict(X_test)
#print(pred)
#print(len(X_train[0]))
acc=accuracy_score(y_test, predictions)
print("Accuracy ",acc*100)
#print(clf.score(x_test,y_test)*100)

#filename = 'finalized_model.sav'
#joblib.dump(clf, filename)
pkl_filename = "features.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(clf, file)
