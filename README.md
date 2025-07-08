# 🔐 Biometric Attendance System (Face Recognition + Location Verification)

A secure, contactless, and real-time biometric attendance system built using Python. This system uses a **hybrid facial recognition model (LBPH + SVC)** and **location verification (GPS/Wi-Fi)** to ensure attendance is marked **only** when the user is physically present. It is designed for use in educational institutions, workplaces, and organizations seeking an automated and tamper-proof attendance solution.

---

## 📌 Key Features

- ✅ **Hybrid Face Recognition:** Uses both LBPH and SVC models for improved accuracy.
- 📍 **Location Verification:** Confirms physical presence using GPS/Wi-Fi before marking attendance.
- 📸 **Live Face Capture:** Captures and stores 100+ images per user for model training.
- 📊 **CSV-Based Records:** Attendance and student details are logged in easily accessible CSV files.
- 🖥️ **User-Friendly Interface:** Built using Python's Tkinter GUI framework.
- 💡 **Lightweight & Offline:** Works in real-time without requiring high-end hardware or internet connectivity.

---

## 🧠 How It Works

1. **Registration**:  
   - User enters ID and name via GUI.
   - Webcam captures and stores face images (~100) for training.

2. **Model Training**:  
   - Two models are trained:  
     - **LBPH** (texture-based)  
     - **SVC** (feature-based classification)  
   - Trained models are saved to disk for future recognition.

3. **Attendance Tracking**:  
   - System captures a live face image.
   - Face is matched using both models.
   - Location (GPS/Wi-Fi) is verified.
   - If either model confirms identity and location is valid → attendance is marked in a CSV file.

---

## 🔧 Technologies Used

- **Python 3.x**
- **OpenCV** for face detection and recognition
- **Tkinter** for GUI
- **NumPy**, **Pandas**, **scikit-learn** for data handling and SVC model
- **Haarcascade** for real-time face detection
- **CSV** for record-keeping

---

## 📂 Project Structure

```bash
Biometric-Attendance/
│
├── TrainingImage/               # Captured face images per user
├── TrainingImageLabel/         # Contains Trainner.yml (LBPH) & SVC model
├── StudentDetails/             # student CSV file
├── Attendance/                 # Daily attendance logs (CSV)
├── haarcascade_frontalface_default.xml  # Face detection model
├── main.py                     # Main application code
├── README.md                   # Project documentation
└── requirements.txt            # Python package dependencies
