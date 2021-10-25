from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from PIL import Image 
from numpy import asarray
from numpy import load
from numpy import expand_dims
from numpy import savez_compressed
from numpy import reshape
from keras.models import load_model
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
import pandas as pd
import numpy as np
import pickle
import cv2

def extract_image(image):
  """
  function to extract faces from the images using MTCNN
  """
  img1 = Image.open(image)            #open the image
  img1 = img1.convert('RGB')          #convert the image to RGB format 
  pixels = asarray(img1)              #convert the image to numpy array
  detector = MTCNN()                  #assign the MTCNN detector
  f = detector.detect_faces(pixels)
  #fetching the (x,y)co-ordinate and (width-->w, height-->h) of the image
  x1,y1,w,h = f[0]['box']             
  x1, y1 = abs(x1), abs(y1)
  x2 = abs(x1+w)
  y2 = abs(y1+h)
  #locate the co-ordinates of face in the image
  store_face = pixels[y1:y2,x1:x2]
  #plt.imshow(store_face)
  image1 = Image.fromarray(store_face,'RGB')    #convert the numpy array to object
  image1 = image1.resize((160,160))             #resize the image
  face_array = asarray(image1)                  #image to array
  return face_array

def embed_images(df):
  """
  iterate through all the entries of the csv and go to each image; 
  extract faces and embeddings and store the pair as a numpy array
  """
  x_test = []
  rejected = []
  n = df.shape[0]
  facenet = FaceNet()
  for i in range(n):
    # if i%50 == 0:
    #   print(i)
    try:
      img1 = df['image1'][i]
      address1 = 'dataset_images' + img1
      face1 = extract_image(address1)
      face1 = np.reshape(face1, (1,160, 160,3))
      embed1 = facenet.embeddings(face1)
      img2 = df['image2'][i]
      address2 = 'dataset_images' + img2
      face2 = extract_image(address2)
      face2 = np.reshape(face2, (1,160, 160,3))
      embed2 = facenet.embeddings(face2)
      pair = np.concatenate((embed1,embed2), axis=-1)
      x_test.append(pair)
    except:
      rejected.append(i)
      #print(i, str('NOT OK'))

    return x_test, rejected

df = pd.read_csv('test.csv')
n = df.shape[0]
testX, rejectedIndexes = embed_images(df)
testX = testX.reshape([n,1024])
filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
resultPred = np.zeros(n)
j = 0
for i in range(n):
  if i in rejectedIndexes:
    resultPred[i] = 1
  else:
    rfc_predict = loaded_model.predict(testX[j,:].reshape([1,1024]))
    resultPred[i] = rfc_predict[0]
    j = j + 1

df['label_pred'] = resultPred.astype(int)
df.to_csv('testPred.csv')
