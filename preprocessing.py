import cv2
import numpy as np
import joblib

IMG_SIZE = 198

# Load encoders & scaler
stone_encoder  = joblib.load("stone_encoder.pkl")
auth_encoder   = joblib.load("auth_encoder.pkl")
origin_encoder = joblib.load("origin_encoder.pkl")
color_encoder  = joblib.load("color_encoder.pkl")
scaler         = joblib.load("scaler.pkl")

def preprocess_image(image_bytes: bytes) -> np.ndarray:#Prepare uploaded gemstone image for CNN model prediction
    """Preprocess image exactly like training"""
    np_img = np.frombuffer(image_bytes, np.uint8)#binary data convert to numpy array
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)#converts the array into a real image

    if img is None:
        raise ValueError("Invalid image")#Check Image Validity

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#Convert Color Format
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))#Matching the training dataset image size.
    img = img / 255.0

    return np.expand_dims(img, axis=0)  # (1, 198, 198, 3)

#Tabular Data Preprocessing
def preprocess_tabular(ri: float, sg: float, hardness: float, color: str) -> np.ndarray:
    """Preprocess numeric + categorical features"""
    color_label = color_encoder.transform([color])[0]
#Create Numeric Feature Array
    tabular = np.array([[ri, sg, hardness]])
    tabular_scaled = scaler.transform(tabular)
#Combine Numeric + Categorical Data
    final_tab = np.hstack([tabular_scaled, [[color_label]]])

    return final_tab.astype(np.float32)
