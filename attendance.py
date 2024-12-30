import tkinter as tk
import cv2
import os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
from tkinter import messagebox
import threading

def track_images_threaded():
    threading.Thread(target=track_images).start()

# Constants for file paths
HARCASCADE_PATH = "haarcascade_frontalface_default.xml"
TRAINING_IMAGE_DIR = "TrainingImage"
STUDENT_DETAILS_FILE = "StudentDetails/StudentDetails.csv"
TRAINER_FILE = "TrainingImageLabel/Trainner.yml"
IMAGES_UNKNOWN_DIR = "ImagesUnknown"
ATTENDANCE_DIR = "Attendance"

# Create the main application window
window = tk.Tk()
window.title("Attendance System")
window.configure(background='pink')

# Configure grid layout
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

# UI Elements
def create_label(text, x, y, width=20, height=2, font_size=25, bg="pink", fg="black"):
    label = tk.Label(window, text=text, bg=bg, fg=fg, width=width, height=height, font=('Times New Roman', font_size, 'bold'))
    label.place(x=x, y=y)
    return label

create_label("RKGIT", 1150, 760, 20, 2, 25, "white", "black")
create_label("ATTENDANCE MANAGEMENT PORTAL", 200, 20, 40, 1, 35, "pink", "black")

lbl_id = create_label("Enter Your College Roll No.", 200, 200)
txt_id = tk.Entry(window, width=30, bg="white", fg="blue", font=('Times New Roman', 15, 'bold'))
txt_id.place(x=250, y=300)

lbl_name = create_label("Enter Your Name", 600, 200)
txt_name = tk.Entry(window, width=30, bg="white", fg="blue", font=('Times New Roman', 15, 'bold'))
txt_name.place(x=650, y=300)

lbl_notification = create_label("NOTIFICATION", 1060, 200)
message = tk.Label(window, text="", bg="white", fg="blue", width=30, height=1, font=('Times New Roman', 15, 'bold'))
message.place(x=1075, y=300)

create_label("ATTENDANCE", 120, 570, 20, 2, 30, "lightgreen", "white")
message2 = tk.Label(window, text="", fg="red", bg="yellow", width=60, height=4, font=('times', 15, 'bold'))
message2.place(x=700, y=570)

# Step Labels
create_label("STEP 1", 240, 375, 20, 2, 20, "pink", "green")
create_label("STEP 2", 645, 375, 20, 2, 20, "pink", "green")
create_label("STEP 3", 1100, 362, 20, 2, 20, "pink", "green")

# Clear input fields
def clear_fields():
    txt_id.delete(0, 'end')
    txt_name.delete(0, 'end')
    message.configure(text="")

# Check if input is a number
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# Capture images for training
def take_images():
    Id = txt_id.get()
    name = txt_name.get()
    
    if not Id or not name:
        message.configure(text="Please enter both ID and Name")
        return

    if not is_number(Id) or not name.isalpha():
        message.configure(text="ID must be numeric and Name must be alphabetical")
        return

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(HARCASCADE_PATH)
    sample_num = 0

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            sample_num += 1
            cv2.imwrite(f"{TRAINING_IMAGE_DIR}/{name}.{Id}.{sample_num}.jpg", gray[y:y + h, x:x + w])
            cv2.imshow('frame', img)

        if cv2.waitKey(100 ) & 0xFF == ord('q'):
            break
        elif sample_num > 60:
            break

    cam.release()
    cv2.destroyAllWindows()
    res = f"Images Saved for ID: {Id} Name: {name}"
    row = [Id, name]
    with open(STUDENT_DETAILS_FILE, 'a+') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    message.configure(text=res)

def train_images():
    recognizer = cv2.face.LBPHFaceRecognizer.create()
    faces, Ids = get_images_and_labels(TRAINING_IMAGE_DIR)
    recognizer.train(faces, np.array(Ids))
    recognizer.save(TRAINER_FILE)
    res = "Images Trained"
    clear_fields()
    message.configure(text=res)
    tk.messagebox.showinfo('Completed', 'Your model has been trained successfully!')

def get_images_and_labels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []

    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids

def track_images():
    recognizer = cv2.face.LBPHFaceRecognizer.create()
    recognizer.read(TRAINER_FILE)
    faceCascade = cv2.CascadeClassifier(HARCASCADE_PATH)
    df = pd.read_csv(STUDENT_DETAILS_FILE)
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)

    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])

            if conf < 50:
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa = df.loc[df['Id'] == Id]['Name'].values
                attendance.loc[len(attendance)] = [Id, aa, date, timeStamp]
            else:
                Id = 'Unknown'

            cv2.putText(im, str(Id), (x, y + h), font, 1, (255, 255, 255), 2)

        attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
        cv2.imshow('im', im)

        if cv2.waitKey(1) == ord('q'):
            break

    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour, Minute, Second = timeStamp.split(":")
    fileName = f"{ATTENDANCE_DIR}/Attendance_{date}_{Hour}-{Minute}-{Second}.csv"
    attendance.to_csv(fileName, index=False)
    cam.release()
    cv2.destroyAllWindows()
    message2.configure(text=attendance)
    message.configure(text="Attendance Taken")
    tk.messagebox.showinfo('Completed', 'Congratulations! Your attendance has been marked successfully for the day!')

def quit_window():
    MsgBox = tk.messagebox.askquestion('Exit Application', 'Are you sure you want to exit the application?', icon='warning')
    if MsgBox == 'yes':
        tk.messagebox.showinfo("Greetings", "This is my miniproject. Have a nice day ahead!")
        window.destroy()

# Buttons
takeImg = tk.Button(window, text="IMAGE CAPTURE BUTTON", command=take_images, fg="white", bg="blue", width=25, height=2, activebackground="pink", font=('Times New Roman', 15, 'bold'))
takeImg.place(x=245, y=425)

trainImg = tk.Button(window, text="MODEL TRAINING BUTTON", command=train_images, fg="white", bg="blue", width=25, height=2, activebackground="pink", font=('Times New Roman', 15, 'bold'))
trainImg.place(x=645, y= 425)

trackImg = tk.Button(window, text="ATTENDANCE MARKING BUTTON", command=track_images, fg="white", bg="blue", width=30, height=3, activebackground="pink", font=('Times New Roman', 15, 'bold'))
trackImg.place(x=1075, y=412)

quitWindow = tk.Button(window, text="QUIT", command=quit_window, fg="white", bg="red", width=10, height=2, activebackground="pink", font=('Times New Roman', 15, 'bold'))
quitWindow.place(x=700, y=735)

window.mainloop()