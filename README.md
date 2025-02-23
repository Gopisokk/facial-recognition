Face Recognition Based Attendance Management System
📌 Overview
This project implements an automated attendance management system using Face Recognition. It captures students' faces, stores their data, and marks attendance based on face detection.

🚀 Features
Face Detection & Recognition using OpenCV
Real-time Attendance Marking
Database Integration for storing student details
User-friendly Interface
Logs & Reports Generation
🛠️ Tech Stack
Programming Language: Python
Libraries Used: OpenCV, NumPy, Pandas
Database: Firebase / SQLite
Framework: Flask / Tkinter (if GUI is implemented)
📂 Project Structure
FaceRecognitionAttendanceSystem/
│── dataset/             # Stores captured images
│── models/              # Trained models for face recognition
│── scripts/             # Python scripts for face detection, training, and attendance logging
│── database/            # Attendance records storage
│── app.py               # Main application file
│── requirements.txt     # Dependencies list
│── README.md            # Project documentation
🔧 Installation & Setup
Clone the repository
git clone https://github.com/your-username/facial-recognition.git
cd facial-recognition
Install dependencies
pip install -r requirements.txt
Run the application
python app.py
🎯 How It Works
Register Students: Capture and store face data.
Face Recognition: The system identifies students from live camera feed.
Mark Attendance: Matches face with database and logs attendance.
