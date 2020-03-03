import cv2
import os
import dlib
import numpy as np
import pickle
import pandas as pd
import datetime
import csv
import face_recognition
from scipy import misc
# from sklearn.metrics import confusion_matrix
# from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


from joblib import dump, load
from PIL import Image




# face_detector = dlib.get_frontal_face_detector()
# pose_predictor_68_point = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# face_encoder = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# def whirldata_face_detectors(img, number_of_times_to_upsample=1):
#     return face_detector(img, number_of_times_to_upsample)

# def whirldata_face_encodings(face_image,num_jitters=1):
#     face_locations = whirldata_face_detectors(face_image)
#     print(face_locations)
#     pose_predictor = pose_predictor_68_point
#     predictors = [pose_predictor(face_image, face_location) for face_location in face_locations]
#     return [np.array(face_encoder.compute_face_descriptor(face_image, predictor, num_jitters)) for predictor in predictors][0]

# def learning():

#     encodings = []
#     names = []

#     train_dir = os.listdir('images')

#     for person in train_dir:
#         pix = os.listdir("images/" + person)
        

#         for person_img in pix:

#             face = misc.imread("images/" + person + "/" + person_img)
#             # face = face_recognition.load_image_file("images/" + person + "/" + person_img)
#             face_enc = whirldata_face_encodings(face)

#             encodings.append(face_enc)
#             names.append(person)
        
#         print('learning')
#     print(names)

#     clf = svm.SVC(gamma='scale')
#     clf.fit(encodings,names)


#     print ('learning complete')
#     dump(clf,'trained.joblib')

# learning()


def learning():

    encodings = []
    names = []

    train_dir = os.listdir('images')

    for person in train_dir:
        pix = os.listdir("images/" + person)
        

        for person_img in pix:

            face = face_recognition.load_image_file("images/" + person + "/" + person_img)
            face_enc = face_recognition.face_encodings(face)[0]

            encodings.append(face_enc)
            names.append(person)
        
        print('learning')

    encodings=np.asarray(encodings)
    names=np.asarray(names)

    # X_train, X_test, y_train, y_test = train_test_split(encodings, names, test_size=0.2, random_state=0)


    # print(y_test)
    # print(X_train.shape,y_train.shape)
    # print(X_test.shape,y_test.shape)
    clf = svm.SVC(gamma='scale')

    # clf.fit(X_train,y_train)
    clf.fit(encodings,names)

    # y_pred = clf.predict(X_test)
    # print(confusion_matrix(y_test, y_pred))
    # akurasi = clf.score(encodings, names)
    scores = cross_val_score(clf, encodings, names, cv=5)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    # akurasi = clf.score(X_test,y_test)
    # print(akurasi)
    print ('learning complete')
    dump(clf,'trained.joblib')

learning()