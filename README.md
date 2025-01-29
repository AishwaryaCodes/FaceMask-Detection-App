### Flask | TensorFlow | OpenCV | Python  

## 📌 Overview  
The **Live Face Mask Detection Web App** is a deep learning-based application that detects whether a person is wearing a mask or not in **real-time video streams and images**.  
It uses **MobileNetV2 for classification** and **OpenCV for face detection**, making it efficient for real-world deployment.

---

## 🚀 Features  
✅ Detects face masks in **real-time video streams** and **uploaded images**  
✅ Built with **Flask** for an interactive web-based UI  
✅ Uses **MobileNetV2 + TensorFlow** for accurate mask classification  
✅ Implements **OpenCV for face detection** to improve model performance  
✅ Supports deployment on **local machines or cloud services**  

---

## 📷 Usage

- Click on the **Start Webcam** button in the web interface to begin live mask detection.
- The app will start the webcam feed and process the live video stream in real-time.
- It will display the face bounding box and the mask detection result (Mask or No Mask) with a confidence score.

---

## 🛠 Technologies Used  
🔹 **Python** – Backend logic  
🔹 **Flask** – Web framework  
🔹 **TensorFlow & Keras** – Deep learning model  
🔹 **OpenCV** – Face detection  
🔹 **HTML, CSS, JavaScript** – Frontend  

---

live-face-mask-detection
```
│── model/                # Contains the trained model
│── static/               # Stores images, CSS files
│── templates/            # HTML templates for Flask
│── app.py                # Main Flask application
│── mask_detector.py      # Mask detection model script
│── face_detector/        # OpenCV-based face detection model
│── requirements.txt      # List of dependencies
│── README.md             # Project documentation
```

---

## 🛠️ Installation  

### **1️⃣ Clone the repository**  
```bash
git clone https://github.com/yourusername/live-face-mask-detection.git
cd live-face-mask-detection
```

### **2️⃣ Install dependencies**
```
pip install -r requirements.txt

```

### **3️⃣ Download the pre-trained model**
Download the trained MobileNetV2 model and place it inside the model/ directory. (Provide a link if hosting the model.)


### **4️⃣ Run the application**
```bash
python app.py
```

### **Final**
Now, visit  http://127.0.0.1:5000/ in your browser to use the app.
