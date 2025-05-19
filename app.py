import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image

# ---------- Setup ----------

# Class labels and their descriptions + emojis
class_info = {
    'dew': ('💧 Dew', 'Moisture condensed from the atmosphere, usually seen in the early morning.'),
    'fogsmog': ('🌫️ Fog/Smog', 'Low visibility due to condensed water vapor or pollution.'),
    'frost': ('❄️ Frost', 'Thin layer of ice forming on surfaces due to cold.'),
    'glaze': ('🧊 Glaze', 'Ice coating objects, formed by freezing rain.'),
    'hail': ('🌨️ Hail', 'Pellets of frozen rain falling in showers.'),
    'lightning': ('⚡ Lightning', 'Flash of light caused by an electrical discharge.'),
    'rain': ('🌧️ Rain', 'Precipitation of water droplets from clouds.'),
    'rainbow': ('🌈 Rainbow', 'Spectrum of light appearing in the sky due to rain and sunlight.'),
    'rime': ('🧊 Rime', 'Frost formed when water droplets freeze on cold surfaces.'),
    'sandstorm': ('🌪️ Sandstorm', 'Strong wind carrying sand or dust.'),
    'snow': ('❄️ Snow', 'Frozen crystalline water falling from clouds.')
}

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="Resnet50.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------- Streamlit UI ----------

st.set_page_config(page_title="Weather Classifier", layout="centered")
st.title("🌦️ Weather Image Classifier")
st.write("Upload a weather-related image, and the AI will try to classify it!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # ---------- Image Processing ----------
    img_array = image.img_to_array(img)
    img_array = cv2.resize(img_array, (224, 224))
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # ---------- Prediction ----------
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class_index = int(np.argmax(output_data, axis=1)[0])
    
    class_labels = list(class_info.keys())
    class_key = class_labels[predicted_class_index]
    class_emoji, class_description = class_info[class_key]

    # ---------- Output Display ----------
    st.markdown("---")
    st.subheader("🧠 Prediction Result:")
    st.markdown(f"""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
        <h2 style="color: #333;">{class_emoji}</h2>
        <p style="font-size: 18px;">{class_description}</p>
    </div>
    """, unsafe_allow_html=True)

    st.success(f"✅ The weather condition is **{class_key.upper()}**.")



