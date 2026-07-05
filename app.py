from flask import Flask, request, jsonify, render_template, send_file
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader

app = Flask(__name__)

# Load model VGG16
MODEL_PATH = "models/model_vgg16.h5"
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    model = None

CLASS_NAMES = ["Glioma Tumor", "Meningioma Tumor", "Tidak Ada Tumor (Normal)", "Pituitary Tumor"]

def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Sistem belum siap. Model prediksi tidak ditemukan."}), 500

    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file yang diunggah"}), 400
    
    file = request.files["file"]
    if file.filename == '':
        return jsonify({"error": "File tidak valid"}), 400

    try:
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)
        
        prediction = model.predict(processed_image)
        predicted_class_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        return jsonify({
            "prediction": CLASS_NAMES[predicted_class_index],
            "confidence": f"{confidence:.2f}%"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/export_pdf", methods=["POST"])
def export_pdf():
    try:
        # Menangkap data dari form frontend
        patient_name = request.form.get("patientName", "Tidak Diberikan")
        patient_id = request.form.get("patientId", "-")
        patient_age = request.form.get("patientAge", "-")
        prediction = request.form.get("prediction", "Belum Dianalisis")
        confidence = request.form.get("confidence", "0%")
        notes = request.form.get("notes", "Tidak ada catatan klinis.")
        file = request.files.get("file")

        # Inisialisasi PDF di memori (RAM)
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        # Menyusun Header Laporan
        p.setFont("Helvetica-Bold", 16)
        p.drawString(50, height - 50, "Laporan Diagnostik MRI - VGG16 NeuroDiagnostics")
        
        # Menyusun Data Pasien
        p.setFont("Helvetica", 12)
        p.drawString(50, height - 90, f"Nama Pasien : {patient_name}")
        p.drawString(50, height - 110, f"ID Rekam Medis: {patient_id}")
        p.drawString(50, height - 130, f"Usia Pasien : {patient_age} Tahun")
        
        # Menyusun Hasil Diagnosis
        p.drawString(50, height - 170, "Hasil Analisis Artificial Intelligence (VGG-16):")
        p.setFont("Helvetica-Bold", 12)
        
        # Warna teks merah jika ada tumor, hijau jika normal
        if "Tidak Ada" in prediction or "Normal" in prediction:
            p.setFillColorRGB(0, 0.6, 0) # Hijau
        else:
            p.setFillColorRGB(0.8, 0, 0) # Merah

        p.drawString(50, height - 190, f"Diagnosis: {prediction}")
        p.setFillColorRGB(0, 0, 0) # Kembali ke hitam
        p.drawString(50, height - 210, f"Tingkat Keyakinan Model: {confidence}")
        
        # Menyusun Catatan Klinis
        p.setFont("Helvetica", 12)
        p.drawString(50, height - 250, "Catatan Observasi Dokter:")
        
        # Memecah teks catatan agar tidak keluar garis (text wrap sederhana)
        textobject = p.beginText(50, height - 270)
        textobject.setFont("Helvetica", 11)
        import textwrap
        wrapped_notes = textwrap.wrap(notes, width=80)
        for line in wrapped_notes:
            textobject.textLine(line)
        p.drawText(textobject)
        
        # Menempelkan Gambar MRI jika ada
        if file:
            img = Image.open(io.BytesIO(file.read()))
            img_reader = ImageReader(img)
            p.drawImage(img_reader, 50, height - 600, width=300, preserveAspectRatio=True)

        p.showPage()
        p.save()
        buffer.seek(0)

        # Mengirimkan file PDF ke pengguna
        return send_file(buffer, as_attachment=True, download_name="Laporan_Diagnostik_MRI.pdf", mimetype="application/pdf")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)