import cv2
import os
import numpy as np
import pickle
import pandas as pd
import csv
import face_recognition

from sklearn import svm
from sklearn.model_selection import cross_val_score

from joblib import dump, load
from PIL import Image

from flask import Flask, render_template, request, Response, redirect, url_for, send_from_directory
app = Flask(__name__)

APP_ROOT=os.path.dirname(os.path.abspath(__file__))

camera = cv2.VideoCapture('videos/video_1.mp4')
namaWajah = "image"
webcamPort = 0
mulaiCapture = False

@app.route("/")
def home():
    hapusCamera()
    return render_template('opsiTambah.html')

def get_frame_capture():
    global namaWajah
    global mulaiCapture
    directory="images/{}".format(namaWajah)
    target=os.path.join(APP_ROOT,directory)
    if not os.path.isdir(target):
        os.mkdir(target)

    camera = cv2.VideoCapture(webcamPort)
    frameCount = 0
    jumlahWajah=0
    color = (0, 0, 255)
    while True:
        ret, frame = camera.read()

        if ret == True and (frameCount%5)==0:
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            gray_small_frame = small_frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(gray_small_frame)
            print(face_locations)

            for (top, right, bottom, left) in face_locations:
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            if mulaiCapture == True:
                try:
                    face_image = frame[top:bottom, left:right]
                    pil_image = Image.fromarray(face_image)
                    filename = "{} {}.jpeg".format(namaWajah,jumlahWajah)
                    destination ="/".join([target, filename])
                    print(filename)
                    pil_image.save(destination)
                    print('save')
                    jumlahWajah+=1
                except:
                    print('kosong')

            imgencode=cv2.imencode('.jpg',frame)[1]
            stringData=imgencode.tostring()
            yield (b'--frame\r\n'
                b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
            print(jumlahWajah)
        frameCount+=1
        if jumlahWajah>=10:
            mulaiCapture = False
            break
    del(camera)
    gambarComplete= cv2.imread("static/Check.jpg")
    imgencode=cv2.imencode('.jpg',gambarComplete)[1]
    stringData=imgencode.tostring()
    yield (b'--frame\r\n'
        b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')

@app.route('/mulaiCapture')
def mulaiCapture():
    global mulaiCapture
    mulaiCapture = True
    return redirect(url_for("tambahKameraCap"))

def hapusCamera():
    global camera
    try:
        del(camera)
    except:
        pass


@app.route('/calc2')
def calc2():
    return Response(get_frame_capture(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/upload",methods=['POST'])
def upload():
    nama = request.form['nama']
    directory='images/'+nama
    print(directory)
    target=os.path.join(APP_ROOT,directory)

    if not os.path.isdir(target):
        os.mkdir(target)

    fileUpload = request.files.getlist("file")
    print(fileUpload)
    i=0
    for file in fileUpload:
        print(file)
        face = face_recognition.load_image_file(file)
        try:
            facelocation = face_recognition.face_locations(face)
            top, right, bottom, left = facelocation[0]
            face_image = face[top:bottom, left:right]
            pil_image = Image.fromarray(face_image)

            filename = file.filename
            print(filename)
            destination ="/".join([target, filename])
            pil_image.save(destination)
            i=i+1
        except:
            print('tidak terdeteksi wajah')

    print("===================================")

    if i==0:
        return render_template('failed.html')
    else:
        return render_template('completed.html', pesan1="Data berhasil diupload",pesan2="Silahkan menuju Training untuk melakukan pembelajaran")




@app.route("/tambahUpload")
def tambahUpload():
    return render_template('tambahUpload.html')

@app.route("/tambahKamera")
def tambahKamera():
    return render_template('tambahKamera.html')

@app.route("/tambahKameraCap")
def tambahKameraCap():
    return render_template('tambahKameraCap.html')

@app.route("/getNamaWajah",methods=['POST'])
def getNamaWajah():
    global namaWajah
    namaWajah=request.form['nama']
    return redirect(url_for("tambahKameraCap"))


@app.route("/train")
def train():
    hapusCamera()
    train_dir = os.listdir('images')
    names=[]
    counts=[]
    for person in train_dir:
        path = "images/" + person
        pix = os.listdir("images/" + person)
        if len(pix)==0:
            os.rmdir(path)
            continue
        names.append(person)
        counts.append(len(pix))
    return render_template('train.html',names=names,count=counts,dataCount=len(names))

@app.route("/learning")
def learning():
    print('Mulai Learning')
    encodings = []
    names = []

    train_dir = os.listdir('images')
    for person in train_dir:
        pix = os.listdir("images/" + person)
        for person_img in pix:

            face = face_recognition.load_image_file("images/" + person + "/" + person_img)
            try:
                face_enc = face_recognition.face_encodings(face)[0]
                encodings.append(face_enc)
                names.append(person)
            except:
                print("Tidak terdapat wajah pada gambar")
        print(person)
    print(names)

    clf = svm.SVC(kernel='rbf',gamma='scale')
    clf.fit(encodings,names)
    akurasi=0

    scores = cross_val_score(clf, encodings, names, cv=5)
    print('RBF',scores)
    print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))

    model1 = svm.SVC(kernel='linear',C=1)
    model1.fit(encodings,names)
    scores1 = cross_val_score(model1, encodings, names, cv=5)
    print('Linear',scores1)
    print("Accuracy: %0.5f (+/- %0.5f)" % (scores1.mean(), scores1.std() * 2))

    model2 = svm.SVC(kernel='sigmoid',gamma='auto')
    model2.fit(encodings,names)
    scores2 = cross_val_score(model2, encodings, names, cv=5)
    print('Sigmoid',scores2)
    print("Accuracy: %0.5f (+/- %0.5f)" % (scores2.mean(), scores2.std() * 2))

    print ('learning complete')
    dump(model1,'trained.joblib')
    return render_template('completed.html', pesan1="Data berhasil di Train",pesan2="")

if __name__ == '__main__':
    app.run(debug=True)
