############################################# IMPORTING ################################################
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mess
import tkinter.simpledialog as tsd
import cv2,os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import os


############################################# FUNCTIONS ################################################

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    

##################################################################################

def tick():
    
    time_string = time.strftime('%H:%M:%S')
    clock.config(text=time_string)
    clock.after(200,tick)


###################################################################################

def check_haarcascadefile():
    exists = os.path.isfile("haarcascade_frontalface_default.xml")
    if exists:
        pass
    else:
        mess._show(title='Some file missing', message='Please contact us for help')
        window.destroy()

###################################################################################

def save_pass():
    assure_path_exists("TrainingImageLabel/")
    exists1 = os.path.isfile("TrainingImageLabel\psd.txt")
    if exists1:
        tf = open("TrainingImageLabel\psd.txt", "r")
        key = tf.read()
    else:
        master.destroy()
        new_pas = tsd.askstring('Old Password not found', 'Please enter a new password below', show='*')
        if new_pas == None:
            mess._show(title='No Password Entered', message='Password not set!! Please try again')
        else:
            tf = open("TrainingImageLabel\psd.txt", "w")
            tf.write(new_pas)
            mess._show(title='Password Registered', message='New password was registered successfully!!')
            return
    op = (old.get())
    newp= (new.get())
    nnewp = (nnew.get())
    print(op)
    if (op == key):
        if(newp == nnewp):
            txf = open("TrainingImageLabel\psd.txt", "w")
            txf.write(newp)
        else:
            mess._show(title='Error', message='Confirm new password again!!!')
            return
    else:
        mess._show(title='Wrong Password', message=op)
        return
    mess._show(title='Password Changed', message='Password changed successfully!!')
    master.destroy()

###################################################################################

def change_pass():
    global master
    master = tk.Tk()
    master.geometry("400x160")
    master.resizable(False,False)
    master.title("Change Password")
    master.configure(background="white")
    lbl4 = tk.Label(master,text='    Enter Old Password',bg='white',font=('times', 12, ' bold '))
    lbl4.place(x=10,y=10)
    global old
    old=tk.Entry(master,width=25 ,fg="black",relief='solid',font=('times', 12, ' bold '),show='*')
    old.place(x=180,y=10)
    lbl5 = tk.Label(master, text='   Enter New Password', bg='white', font=('times', 12, ' bold '))
    lbl5.place(x=10, y=45)
    global new
    new = tk.Entry(master, width=25, fg="black",relief='solid', font=('times', 12, ' bold '),show='*')
    new.place(x=180, y=45)
    lbl6 = tk.Label(master, text='Confirm New Password', bg='white', font=('times', 12, ' bold '))
    lbl6.place(x=10, y=80)
    global nnew
    nnew = tk.Entry(master, width=25, fg="black", relief='solid',font=('times', 12, ' bold '),show='*')
    nnew.place(x=180, y=80)
    cancel=tk.Button(master,text="Cancel", command=master.destroy ,fg="black"  ,bg="red" ,height=1,width=25 , activebackground = "white" ,font=('times', 10, ' bold '))
    cancel.place(x=200, y=120)
    save1 = tk.Button(master, text="Save", command=save_pass, fg="black", bg="#3ece48", height = 1,width=25, activebackground="white", font=('times', 10, ' bold '))
    save1.place(x=10, y=120)
    master.mainloop()

#####################################################################################

#verifying password for training image

def psw():
    
    assure_path_exists("TrainingImageLabel/")
    exists1 = os.path.isfile("TrainingImageLabel\psd.txt")
    if exists1:
        tf = open("TrainingImageLabel\psd.txt", "r")
        key = tf.read()
    else:
        new_pas = tsd.askstring('Old Password not found', 'Please enter a new password below', show='*')
        if new_pas == None:
            mess._show(title='No Password Entered', message='Password not set!! Please try again')
        else:
            tf = open("TrainingImageLabel\psd.txt", "w")
            tf.write(new_pas)
            mess._show(title='Password Registered', message='New password was registered successfully!!')
            return
    password = tsd.askstring('Password', 'Enter Password', show='*')
    if (password == key):
        TrainImages()
    elif (password == None):
        pass
    else:
        mess._show(title='Wrong Password', message="you have entered worng password")

######################################################################################

def clear():
    txt.delete(0, 'end')
    res = "1)Take Images  >>>  2)Save Profile"
    message1.configure(text=res)


def clear2():
    txt2.delete(0, 'end')
    res = "1)Take Images  >>>  2)Save Profile"
    message1.configure(text=res)

#######################################################################################
def TakeImages():
    check_haarcascadefile()
    columns = ['SERIAL NO.', '', 'ID', '', 'NAME']
    assure_path_exists("StudentDetails/")
    assure_path_exists("TrainingImage/")
    serial = 0
    exists = os.path.isfile("StudentDetails\StudentDetails.csv")
    if exists:
        with open("StudentDetails\StudentDetails.csv", 'r') as csvFile1:
            reader1 = csv.reader(csvFile1)
            for l in reader1:
                serial = serial + 1
        serial = (serial // 2)
        csvFile1.close()
    else:
        with open("StudentDetails\StudentDetails.csv", 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(columns)
            serial = 1
        csvFile1.close()
    Id = (txt.get())
    name = (txt2.get())
    if ((name.isalpha()) or (' ' in name)):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        if detector.empty():
            print("Failed to load Haar Cascade XML")
            return

        sampleNum = 0
        while (True):
            ret, img = cam.read()
            if not ret:
                print("Failed to capture frame from camera. Exiting...")
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # incrementing sample number
                sampleNum = sampleNum + 1
                # saving the captured face in the dataset folder TrainingImage
                cv2.imwrite(f"TrainingImage/{name}.{serial}.{Id}.{sampleNum}.jpg", gray[y:y+h, x:x+w])

                # display the frame
                cv2.imshow('Taking Images', img)
            # wait for 100 miliseconds
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum > 100:
                break
        cam.release()
        cv2.destroyAllWindows()
        res = "Images Taken for ID : " + Id
        row = [serial, '', Id, '', name]
        with open('StudentDetails\StudentDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message1.configure(text=res)
    else:
        if (name.isalpha() == False):
            res = "Enter Correct name"
            message.configure(text=res)

def getImagesAndLabels(path):
 
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
  
    faces = []
 
    Ids = []
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')
        # getting the Id from the image
        ID = int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(ID)
    return faces, Ids

def contact():
    pass




from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Update TrainImages to include SVC model training
def TrainImages():
    check_haarcascadefile()
    assure_path_exists("TrainingImageLabel/")
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)

    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, ID = getImagesAndLabels("TrainingImage")

    try:
        recognizer.train(faces, np.array(ID))
        recognizer.save("TrainingImageLabel/Trainner.yml")
    except:
        mess._show(title='No Registrations', message='Please Register someone first!!!')
        return

    # SVC model training
    X, y = [], []
    for face, label in zip(faces, ID):
        resized = cv2.resize(face, (100, 100))
        X.append(resized.flatten())
        y.append(label)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    svc_model = SVC(kernel='linear', probability=True)
    svc_model.fit(X_train, y_train)
    acc = accuracy_score(y_test, svc_model.predict(X_test))
    joblib.dump(svc_model, "TrainingImageLabel/svc_model.pkl")

    res = "Models Trained Successfully (SVC Accuracy: {:.2f}%)".format(acc * 100)
    message1.configure(text=res)
    message.configure(text='Total Registrations till now  : ' + str(len(set(ID))))
#####################################################################################################
def show_attendance_popup(message, image_path):
    popup = tk.Toplevel()
    popup.title("Attendance Registered")
    popup.configure(bg="white")

    # Load and display image
    try:
        img = Image.open(image_path)
        img = img.resize((200, 200))
        photo = ImageTk.PhotoImage(img)
        img_label = tk.Label(popup, image=photo, bg="white")
        img_label.image = photo  # Keep a reference
        img_label.pack(pady=10)
    except Exception as e:
        print(f"Image load error: {e}")

    # Attendance message
    msg_label = tk.Label(popup, text=message, font=("Segoe UI", 14), bg="white", fg="green")
    msg_label.pack(pady=10)

    # OK Button
    ok_button = tk.Button(popup, text="OK", command=popup.destroy, font=("Segoe UI", 12), bg="#2ecc71", fg="white")
    ok_button.pack(pady=10)

    popup.update_idletasks()

    # ➡️ Center the popup on the screen
    window_width = 400
    window_height = 370
    screen_width = popup.winfo_screenwidth()
    screen_height = popup.winfo_screenheight()

    x = (screen_width // 2) - (window_width // 2)
    y = (screen_height // 2) - (window_height // 2)

    popup.geometry(f"{window_width}x{window_height}+{x}+{y}")

    popup.grab_set()  # Make it modal


##################################################################################################
"""from pyfingerprint.pyfingerprint import PyFingerprint
def fingerprintverification(fingerstring):
    try:
        # Initialize the fingerprint sensor (adjust the serial port as needed)
        f = PyFingerprint('/dev/ttyUSB0', 57600)

        if f.verifyPassword() is False:
            raise ValueError('Fingerprint sensor password is incorrect!')

    except Exception as e:
        print('Failed to initialize sensor!')
        print('Exception message: ' + str(e))
        exit(1)

    print('Waiting for finger...')

    # Wait for finger
    while f.readImage() is False:
        pass

    # Convert image to characteristics and store in charbuffer 1
    f.convertImage(0x01)

    # Search for a fingerprint template in the database
    result = f.searchTemplate()

    positionNumber = result[0]

    if positionNumber >= 0:
        print(f'Fingerprint matched at position #{positionNumber}')
    else:
        print('No match found.')
"""
####################################################################################################

def TrackImages():
    # Check location/WiFi status
    mesg, isstate = Tracklocation()
    if not isstate:
        mess._show(title='Attendance Request Denied!', message=mesg)
        return

    check_haarcascadefile()
    assure_path_exists("Attendance/")
    assure_path_exists("StudentDetails/")
    os.makedirs("Liveimages", exist_ok=True)  # Ensure Liveimages folder exists

    # Clear previous entries in the table
    for k in tv.get_children():
        tv.delete(k)

    # Load face recognizer and Haar cascade
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    if os.path.isfile("TrainingImageLabel/Trainner.yml"):
        recognizer.read("TrainingImageLabel/Trainner.yml")
    else:
        mess._show(title='Data Missing', message='Please click on Save Profile to reset data!!')
        return

    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)

    # Open camera
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', '', 'Name', '', 'Date', '', 'Time']

    # Load student details
    if os.path.isfile("StudentDetails/StudentDetails.csv"):
        df = pd.read_csv("StudentDetails/StudentDetails.csv")
    else:
        mess._show(title='Details Missing', message='Students details are missing, please check!')
        cam.release()
        cv2.destroyAllWindows()
        window.destroy()
        return

    # Define variables
    attendance = ""
    live_image_path = os.path.join("Liveimages", "live_face.jpg")

    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (225, 0, 0), 2)

            # Predict using gray image for better accuracy
            serial, conf = recognizer.predict(gray[y:y + h, x:x + w])

            if conf < 50:
                # Crop color face image
                        # Add padding to capture full head and ears
                padding = 40  # or 40 if you want even bigger
                x1 = max(x - padding, 0)
                y1 = max(y - padding, 0)
                x2 = min(x + w + padding, frame.shape[1])
                y2 = min(y + h + padding, frame.shape[0])

                # Save expanded face image
                color_face_img = frame[y1:y2, x1:x2]
            
                live_image_path = os.path.join("Liveimages", "live_face.jpg")
                cv2.imwrite(live_image_path, color_face_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                # Get date and time
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

                # Fetch student details
                aa = df.loc[df['SERIAL NO.'] == serial]['NAME'].values
                ID = df.loc[df['SERIAL NO.'] == serial]['ID'].values
                ID = str(ID)[1:-1]
                name = str(aa)[2:-2]

                # Check attendance source (WiFi or GPS)
                source = "WIFI" if "wifi" in mesg.lower() else "GPS"
                attendance = [str(ID), '', name, '', date, '', timeStamp, '', source]

                # Display name on video feed
                cv2.putText(frame, name, (x, y + h), font, 1, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "Unknown", (x, y + h), font, 1, (255, 255, 255), 2)

        cv2.imshow('Taking Attendance', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) == ord('q'):
            break

    # Cleanup camera and windows
    cam.release()
    cv2.destroyAllWindows()

    if not attendance:
        mess._show(title='No Face Detected', message='No attendance recorded!')
        return

    # Prepare attendance file
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
    attendance_file = os.path.join("Attendance", f"Attendance_{date}.csv")

    alreadyexist = False
    # Check if attendance already exists
    if os.path.isfile(attendance_file):
        with open(attendance_file, 'r') as csvFile1:
            reader1 = csv.reader(csvFile1)
            next(reader1, None)  # Skip header
            for lines in reader1:
                if len(lines) < 7:
                    continue
                if lines[0] == attendance[0] and lines[2] == attendance[2]:
                    mess._show(title='Attendance Marked in Past', message=f"Attendance already marked for ID: {lines[0]}, Name: {lines[2].upper()}")
                    alreadyexist = True
                    break

    # Save new attendance if not already marked
    if not alreadyexist:
        with open(attendance_file, 'a+', newline='') as csvFile1:
            writer = csv.writer(csvFile1)
            if os.stat(attendance_file).st_size == 0:
                writer.writerow(col_names)
            writer.writerow(attendance)

    # Update GUI table
    with open(attendance_file, 'r') as csvFile1:
        reader1 = csv.reader(csvFile1)
        next(reader1, None)
        for lines in reader1:
            if len(lines) < 7:
                continue
            tv.insert('', 'end', text=lines[0], values=(lines[2], lines[4], lines[6]))

    # Show success popup if attendance marked
    if not alreadyexist:
        show_attendance_popup(f"Good Morning, {attendance[2]}!\nAttendance marked via {attendance[-1]}.", live_image_path)

###########################################################################################
import socket
def checkradius(latitude,longitude):
    from math import radians, cos, sin, asin, sqrt
    classcordinates=[17.3845678,78.4563723]
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000  # Radius of Earth in meters
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        # Haversine formula
        dlat = lat2 - lat1 
        dlon = lon2 - lon1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        distance = R * c
        return distance
    
    current_lat = latitude
    current_lon = longitude
    lat = latitude
    lon = longitude

    # Calculate the distance
    distance = haversine(current_lat, current_lon,lat,lon)/100
    if distance <= 50:
        return True
    else:
        return False

#Trackig location and Wifi network
def Tracklocation():
    
    desried="192.168.153"
    hostname = socket.gethostname()
    local_ip = str(socket.gethostbyname(hostname))
    isstart=local_ip.startswith(desried)
    if not isstart:
        return ("Attendence marked Through Wifi connection",True)
    
    import requests
    try:
        response = requests.get('https://ipinfo.io/json')
        data = response.json()
        loc = data.get('loc') 
        if loc:
            latitude, longitude =map(float, loc.split(','))
            if checkradius(latitude,longitude):
                return ("Attendence marked Through Location",True)
            else:
                return ("You are not in desired Range from Department",False)
            
        else:
            return ("Location not found",False)
            
            

    except Exception as e:
        return ("Connection Error",False)
    
    return False

    

######################################## USED STUFFS ############################################
    
global key
key = ''

ts = time.time()
date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
day,month,year=date.split("-")

mont={'01':'January',
      '02':'February',
      '03':'March',
      '04':'April',
      '05':'May',
      '06':'June',
      '07':'July',
      '08':'August',
      '09':'September',
      '10':'October',
      '11':'November',
      '12':'December'
      }

######################################## GUI FRONT-END ###########################################

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mess
import tkinter.simpledialog as tsd
import cv2, os, csv, numpy as np
from PIL import Image
import pandas as pd
import datetime, time
######################################## GUI FRONT-END ###########################################

# ---------- Main Window ----------
window = tk.Tk()
window.geometry("1280x720")
window.resizable(True, False)
window.title("Attendance System")
window.configure(background="#1e1e2f")  # Dark background

# ---------- Styles for ttk Treeview ----------
style = ttk.Style()
style.theme_use('clam')
style.configure("Treeview.Heading", background="#34495e", foreground="white", font=("Segoe UI", 12, "bold"))
style.configure("Treeview", font=("Segoe UI", 11), rowheight=26, background="#ecf0f3", fieldbackground="#ecf0f3")

# ---------- Header ----------
message3 = tk.Label(window,
    text="Face Recognition Based Attendance System",
    fg="white", bg="#1e1e2f",
    font=("Segoe UI", 24, "bold")
)
message3.place(x=10, y=10)

# ---------- Date & Clock Frames ----------
ts = time.time()
date = datetime.datetime.fromtimestamp(ts).strftime('%d-%B-%Y')
day, month, year = date.split('-')[0], date.split('-')[1], date.split('-')[2]
# month name already full

frame_date = tk.Frame(window, bg="#2c3e50", bd=0)
frame_date.place(relx=0.36, rely=0.09, relwidth=0.16, relheight=0.07)

frame_clock = tk.Frame(window, bg="#2c3e50", bd=0)
frame_clock.place(relx=0.52, rely=0.09, relwidth=0.09, relheight=0.07)

datef = tk.Label(
    frame_date,
    text=f"{day}-{month}-{year}",
    fg="#f1c40f", bg="#2c3e50",
    font=("Segoe UI", 14, "bold")
)
datef.pack(fill='both', expand=True)

clock = tk.Label(
    frame_clock,
    fg="#f1c40f", bg="#2c3e50",
    font=("Segoe UI", 14, "bold")
)
clock.pack(fill='both', expand=True)

def tick():
    clock.config(text=time.strftime('%H:%M:%S'))
    clock.after(1000, tick)

tick()

# ---------- Main Frames ----------
frame1 = tk.Frame(window, bg="#2c3e50", bd=2, relief="ridge")
frame1.place(relx=0.11, rely=0.17, relwidth=0.39, relheight=0.80)

frame2 = tk.Frame(window, bg="#2c3e50", bd=2, relief="ridge")
frame2.place(relx=0.51, rely=0.17, relwidth=0.38, relheight=0.80)

# ---------- Section Headers ----------
head1 = tk.Label(
    frame1,
    text="For Already Registered",
    fg="white", bg="#2ecc71",
    font=("Segoe UI", 16, "bold")
)
head1.place(x=0, y=0, relwidth=1)

head2 = tk.Label(
    frame2,
    text="For New Registrations",
    fg="white", bg="#2ecc71",
    font=("Segoe UI", 16, "bold")
)
head2.grid(row=0, column=0, columnspan=2, sticky="ew")

# ---------- Registration Fields ----------
tk.Label(
    frame2,
    text="Enter ID",
    fg="white", bg="#2c3e50",
    font=("Segoe UI", 14)
).place(x=80, y=55)
txt = tk.Entry(
    frame2,
    width=32,
    font=("Segoe UI", 13),
    bg="#ecf0f3", relief="flat", bd=2
)
txt.place(x=30, y=88)

clearButton = tk.Button(
    frame2,
    text="Clear",
    command=clear,
    fg="white", bg="#e74c3c",
    activebackground="#c0392b",
    width=11,
    font=("Segoe UI", 11, "bold"),
    relief="flat"
)
clearButton.place(x=335, y=86)

# ---------- Name Field ----------
tk.Label(
    frame2,
    text="Enter Name",
    fg="white", bg="#2c3e50",
    font=("Segoe UI", 14)
).place(x=80, y=140)
txt2 = tk.Entry(
    frame2,
    width=32,
    font=("Segoe UI", 13),
    bg="#ecf0f3", relief="flat", bd=2
)
txt2.place(x=30, y=173)

clearButton2 = tk.Button(
    frame2,
    text="Clear",
    command=clear2,
    fg="white", bg="#e74c3c",
    activebackground="#c0392b",
    width=11,
    font=("Segoe UI", 11, "bold"),
    relief="flat"
)
clearButton2.place(x=335, y=172)

# ---------- Instruction & Status ----------
message1 = tk.Label(
    frame2,
    text="1)Take Images  >>>  2)Save Profile",
    bg="#2c3e50", fg="white",
    font=("Segoe UI", 12, "bold")
)
message1.place(x=30, y=230)

message = tk.Label(
    frame2,
    text="",
    bg="#2c3e50", fg="#f1c40f",
    font=("Segoe UI", 12, "bold")
)
message.place(x=30, y=450)

# ---------- Action Buttons ----------
takeImg = tk.Button(
    frame2,
    text="Take Images",
    command=TakeImages,
    fg="white", bg="#2980b9",
    activebackground="#3498db",
    width=34, height=1,
    font=("Segoe UI", 13, "bold"),
    relief="flat"
)
takeImg.place(x=30, y=400)




trainImg = tk.Button(
    frame2,
    text="Save Profile",
    command=psw,
    fg="white", bg="#2980b9",
    activebackground="#3498db",
    width=34, height=1,
    font=("Segoe UI", 13, "bold"),
    relief="flat"
)
trainImg.place(x=30, y=480)

# ---------- Attendance Section ----------
trackImg = tk.Button(
    frame1,
    text="Take Attendance",
    command=TrackImages,
    fg="white", bg="#f39c12",
    activebackground="#f1c40f",
    width=35, height=1,
    font=("Segoe UI", 13, "bold"),
    relief="flat"
)
trackImg.place(x=30, y=50)

lbl3 = tk.Label(
    frame1,
    text="Attendance",
    fg="white", bg="#2c3e50",
    font=("Segoe UI", 14, "bold")
)
lbl3.place(x=100, y=115)

quitWindow = tk.Button(
    frame1,
    text="Quit",
    command=window.destroy,
    fg="white", bg="#e74c3c",
    activebackground="#c0392b",
    width=35, height=1,
    font=("Segoe UI", 13, "bold"),
    relief="flat"
)
quitWindow.place(x=30, y=450)

# ---------- Menubar ----------
menubar = tk.Menu(window, relief='ridge')
filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label='Change Password', command=change_pass)
filemenu.add_command(label='Contact Us', command=contact)
filemenu.add_command(label='Exit', command=window.destroy)
menubar.add_cascade(label='Help', menu=filemenu)
window.configure(menu=menubar)

# ---------- Attendance Table ----------
tv = ttk.Treeview(frame1, height=13, columns=('name','date','time'))
tv.column('#0', width=82)
tv.column('name', width=130)
tv.column('date', width=133)
tv.column('time', width=133)
tv.heading('#0', text='ID')
tv.heading('name', text='NAME')
tv.heading('date', text='DATE')
tv.heading('time', text='TIME')
tv.grid(row=2, column=0, padx=(0,0), pady=(150,0), columnspan=4)

scroll=ttk.Scrollbar(frame1, orient='vertical', command=tv.yview)
scroll.grid(row=2, column=4, padx=(0,100), pady=(150,0), sticky='ns')
tv.configure(yscrollcommand=scroll.set)

window.mainloop()
