import cv2
import os
from flask import Flask, request, render_template
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import pickle

app = Flask(__name__)

nimgs = 10

imgBackground=cv2.imread("background.png")

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

def totalreg():
    return len(os.listdir('static/faces'))

def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []

def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l

def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')

def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l


@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/start', methods=['GET'])
def start():
    facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    with open('data/faces.pkl', 'rb') as w:
        faces = pickle.load(w)
    with open('data/names.pkl', 'rb') as file:
        labels = pickle.load(file)
    camera = cv2.VideoCapture(0)
    print('Shape of Faces matrix --> ', faces.shape)
    knn = KNeighborsClassifier(n_neighbors=4)
    knn.fit(faces,labels)
    while True:
        ret, frame = camera.read()
        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_coordinates = facecascade.detectMultiScale(gray, 1.3, 5)

            for (a, b, w, h) in face_coordinates:
                fc = frame[b:b + h, a:a + w, :]
                r = cv2.resize(fc, (50, 50)).flatten().reshape(1,-1)
                text = knn.predict(r)
                add_attendance(text[0])
                cv2.putText(frame, text[0], (a, b-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.rectangle(frame, (a, b), (a + w, b + w), (0, 0, 255), 2)

            cv2.imshow('Attendance', frame)
            if cv2.waitKey(1) & 0xFF == ord('o'):
                break
        else:
            print("error")
            break

    cv2.destroyAllWindows()
    camera.release()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)



@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    cname= newusername+'_'+newuserid
    face_data = []
    i = 0
    camera = cv2.VideoCapture(0)
    facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    name = cname
    ret = True
    while(ret):
        ret, frame = camera.read()
        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            face_coordinates = facecascade.detectMultiScale(gray, 1.3, 4)

            for (a, b, w, h) in face_coordinates:
                faces = frame[b:b+h, a:a+w, :]
                resized_faces = cv2.resize(faces, (50, 50))
            
                if i % 10 == 0 and len(face_data) < 10:
                    face_data.append(resized_faces)
                cv2.rectangle(frame, (a, b), (a+w, b+h), (255, 0, 0), 2)
            i += 1

            cv2.imshow('frames', frame)

            if cv2.waitKey(1) == 27 or len(face_data) >= 10:
                break
        else:
            print('error')
            break

    cv2.destroyAllWindows()
    camera.release()


    face_data = np.asarray(face_data)
    face_data = face_data.reshape(10, -1)

    if 'names.pkl' not in os.listdir('data/'):
        names = [name]*10
        with open('data/names.pkl', 'wb') as file:
            pickle.dump(names, file)
    else:
        with open('data/names.pkl', 'rb') as file:
            names = pickle.load(file)

        names = names + [name]*10
        with open('data/names.pkl', 'wb') as file:
            pickle.dump(names, file)


    if 'faces.pkl' not in os.listdir('data/'):
        with open('data/faces.pkl', 'wb') as w:
            pickle.dump(face_data, w)
    else:
        with open('data/faces.pkl', 'rb') as w:
            faces = pickle.load(w)

        faces = np.append(faces, face_data, axis=0)
        with open('data/faces.pkl', 'wb') as w:
            pickle.dump(faces, w)
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

if __name__ == '__main__':
    app.run(debug=True)
