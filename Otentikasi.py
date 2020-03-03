import cv2
import os
import numpy as np
import pickle
import pandas as pd
import datetime
import csv
import face_recognition
import time

from datetime import date
from joblib import dump, load
from PIL import Image

from flask import Flask, render_template, request, Response, redirect, url_for, send_from_directory
app = Flask(__name__)

APP_ROOT=os.path.dirname(os.path.abspath(__file__))

camera = cv2.VideoCapture('videos/video_1.mp4')
namaWajah = "a"
webcamPort = 0

@app.route("/")
def home():
    video_dir = os.listdir('videos')
    print(video_dir)
    names=[]
    for vid in video_dir:
        names.append(vid)


    return render_template('home.html',vid=names)

def get_frame():
    global statusVideo
    global camera
    try:
        camera
    except:
        camera=cv2.VideoCapture('videos/video_1.mp4')
    model=load('trained.joblib')

    waktuEksekusi=0
    frameCount=0
    validationCount=0
    face_temp=[]
    lastWrite=[]
    frameAdaWajah=0
    color=(0, 0, 255) #merah
    while True:
        ret, frame = camera.read()

        if ret == True and (frameCount%5)==0:
            start = time.time()
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = small_frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names=[]
            
            for face_encoding in face_encodings:

                id = model.predict([face_encoding])
                name=id[0]

                face_names.append(name)
            # print(face_temp,face_names)
            if(face_temp==face_names):
                validationCount+=1
                # print(validationCount)
            else:
                color=(0,0,255)
                validationCount=0
                face_temp=face_names

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2


                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)


                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, str(name), (left + 6, bottom - 6), font, 0.75, (0, 0, 0),1)
            
            
            # Tampilkan Gambar

            imgencode=cv2.imencode('.jpg',frame)[1]
            stringData=imgencode.tostring()
            yield (b'--frame\r\n'
                b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
            
            if face_names!=[]:
                print('ada wajah')
                waktuExek = float(time.time()-start)
                waktuEksekusi+=waktuExek
                frameAdaWajah+=1

            if len(face_names)>=1:
                if(validationCount==5):
                    validationCount=0
                    if(lastWrite!=face_names):
                        waktu = datetime.datetime.now()
                        row = [face_names[0],waktu.strftime("%d-%B-%y"),waktu.strftime("%X")]
                        with open('absen.csv', 'a',newline='') as csvFile:
                            writer = csv.writer(csvFile)
                            writer.writerow(row)
                        csvFile.close()

                        print('DITULIS')
                        lastWrite=face_names
                        print(lastWrite)
                        color=(0,255,0)
            try:
                print('rata-rata: ',waktuEksekusi/frameAdaWajah)
            except:
                print('skip')
        frameCount=frameCount+1

        if cv2.waitKey(10) & 0xFF==ord('q'):
            break
    print('BREAK')
    del(camera)

def hapusCamera():
    global camera
    try:
        del(camera)
    except:
        pass
@app.route("/gantiVideo",methods=['POST'])
def gantiVideo():

    status=request.form['isi']
    global camera
    print(status)

    if status=="webcam":
        camera = cv2.VideoCapture(webcamPort)
    # elif status=="webcam2":
    #     camera = cv2.VideoCapture(1)
    else:
        camera = cv2.VideoCapture("videos/{}".format(status))
    return redirect(url_for('home_play'))

@app.route("/home_play")
def home_play():
    return render_template('home_play.html')

@app.route('/calc')
def calc():
    return Response(get_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/absensi")
def absensi():
    hapusCamera()
    id_list = []
    dates = []
    times = []
    # absen=
    try:
        absen = pd.read_csv("absen.csv",names=["Id","Date","Time"])
        id_list = absen["Id"].tolist()
        dates = absen["Date"].tolist()
        times = absen["Time"].tolist()

        id_list = id_list[::-1]
        dates = dates[::-1]
        times = times[::-1]
    except:
        with open("absen.csv", "w") as my_empty_csv:
            pass
        print('Tidak ada csv')
    pelaporan()
    return render_template('absensi.html',ids=id_list,dates=dates,times=times,dataCount=len(id_list))

def pelaporan():
    tanggalUnik=[]
    try:
        absen = pd.read_csv("absen.csv",names=["Id","Date","Time"])
        print(absen)
        tanggalUnik = absen["Date"].unique().tolist()
        print(tanggalUnik)
    except:
        print('Absen Kosong')

    try:
        os.remove('laporan.csv')
    except:
        print("masih kosong")
    with open('laporan.csv', 'w',newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(["Tanggal","Mahasiswa yang Hadir","Jumlah Mahasiswa"])
        for tanggal in tanggalUnik:
            absenTanggal = absen[absen.Date==tanggal]
            orangUnik = absenTanggal["Id"].unique().tolist()
            row = [tanggal, orangUnik, len(orangUnik)]
            
            writer.writerow(row)
        csvFile.close()

def pelaporan2():

    tanggalUnik=[]
    # try:
    absen = pd.read_csv("absen.csv",names=["Id","Date","Time"])
    print(absen)
    tanggalUnik = absen["Date"].unique().tolist()
    print(tanggalUnik)
    print(absen[(absen['Date'] > '2019-08-01') & (absen['Date'] < '2013-09-01')])
    # except:
        # print('Absen Kosong')

    try:
        os.remove('laporan.csv')
    except:
        print("masih kosong")
    with open('laporan.csv', 'w',newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(["Tanggal","Mahasiswa yang Hadir","Jumlah Mahasiswa"])
        for tanggal in tanggalUnik:
            absenTanggal = absen[absen.Date==tanggal]
            orangUnik = absenTanggal["Id"].unique().tolist()
            row = [tanggal, orangUnik, len(orangUnik)]
            
            writer.writerow(row)
        csvFile.close()

@app.route("/get_csv")
def get_csv():
    filename = "laporan.csv"
    try:
        return send_from_directory(directory=APP_ROOT, filename=filename, as_attachment=True)
    except FileNotFoundError:
        abort(404)


if __name__ == '__main__':
    app.run(debug=True)
