from flask import Flask, request, render_template, jsonify
import os
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Configuración de rutas
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Carga del modelo YOLO
model = YOLO("best.pt")

# Definir colores por clase
CLASES_COLORS = {
    'GBF': (0, 255, 0),       # Bien fermentado
    'GIF': (0, 255, 255),     # Insuficientemente fermentado
    'GSF': (0, 0, 255),       # Sin fermentar
}

# Verificar si las etiquetas están sincronizadas
MODEL_NAMES = model.names
if not all(clase in MODEL_NAMES.values() for clase in CLASES_COLORS.keys()):
    raise ValueError("Las etiquetas del modelo no coinciden con las definidas en CLASES_COLORS.")

# Filtrar detecciones superpuestas
def filtrar_detecciones(detections, umbral_distancia=50):
    seleccionadas = []
    for i, det1 in enumerate(detections):
        keep = True
        for j, det2 in enumerate(detections):
            if i != j:
                centroide1 = ((det1[0] + det1[2]) / 2, (det1[1] + det1[3]) / 2)
                centroide2 = ((det2[0] + det2[2]) / 2, (det2[1] + det2[3]) / 2)
                distancia = np.sqrt((centroide1[0] - centroide2[0]) ** 2 + (centroide1[1] - centroide2[1]) ** 2)
                if distancia < umbral_distancia and det1[4] < det2[4]:
                    keep = False
                    break
        if keep:
            seleccionadas.append(det1)
    return seleccionadas

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Procesar la imagen subida
        if "file" not in request.files:
            return jsonify({"error": "No se envió ningún archivo"}), 400
        file = request.files["file"]

        # Verificar y guardar archivo
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)

        # Realizar la inferencia
        results = model.predict(source=image_path, conf=0.2, save=False, show=False)
        detections = results[0].boxes
        boxes = detections.xyxy.cpu().numpy()
        confidences = detections.conf.cpu().numpy()
        class_ids = detections.cls.cpu().numpy()

        detecciones = [
            [*boxes[i], confidences[i], int(class_ids[i])] for i in range(len(boxes))
        ]
        detecciones_filtradas = filtrar_detecciones(detecciones)

        # Dibujar detecciones
        img = cv2.imread(image_path)
        for det in detecciones_filtradas:
            x1, y1, x2, y2, _, class_id = det
            class_name = MODEL_NAMES[int(class_id)]
            color = CLASES_COLORS.get(class_name, (255, 0, 0))
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(img, class_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Convertir imagen a base64
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Contar las clases detectadas
        class_counts = {}
        for det in detecciones_filtradas:
            _, _, _, _, _, class_id = det
            class_name = MODEL_NAMES[int(class_id)]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        total_detections = sum(class_counts.values())
        class_percentages = {k: (v / total_detections) * 100 for k, v in class_counts.items()}

        return jsonify({
            "image": img_base64,
            "class_counts": class_counts,
            "class_percentages": class_percentages
        })

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
