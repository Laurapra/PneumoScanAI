import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os
from pathlib import Path

def load_model():
    """Cargar modelo entrenado"""
    print(" Cargando modelo PneumoScan AI...")
    
    custom_objects = {
        'mse': tf.keras.losses.MeanSquaredError(),
        'mae': tf.keras.metrics.MeanAbsoluteError(),
        'binary_crossentropy': tf.keras.losses.BinaryCrossentropy(),
        'sparse_categorical_crossentropy': tf.keras.losses.SparseCategoricalCrossentropy(),
        'accuracy': tf.keras.metrics.Accuracy()
    }
    
    model = tf.keras.models.load_model(
        'models/cpu_local/best_model_cpu.h5',
        custom_objects=custom_objects,
        compile=False
    )
    
    # Re-compilar
    model.compile(
        optimizer='adam',
        loss={
            'detection_output': 'binary_crossentropy',
            'type_output': 'sparse_categorical_crossentropy', 
            'severity_output': 'sparse_categorical_crossentropy',
            'triage_output': 'mse'
        }
    )
    
    print(" Modelo cargado exitosamente!")
    return model

def test_with_real_image(model, image_path, case_type="normal"):
    """Probar con imagen real del dataset"""
    
    print(f"\n Analizando imagen: {Path(image_path).name}")
    
    # Verificar que la imagen existe
    if not os.path.exists(image_path):
        print(f" Imagen no encontrada: {image_path}")
        return None
    
    # Cargar y procesar imagen
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Crear datos clínicos realistas según el tipo de caso
    if case_type == "normal":
        # Paciente normal
        clinical_data = np.array([[0.4, 0, 0, 0, 0.88, 0.5, 0.4, 0.98]])
    elif case_type == "pneumonia_mild":
        # Neumonía leve
        clinical_data = np.array([[0.5, 1, 1, 0, 0.91, 0.65, 0.55, 0.96]])
    elif case_type == "pneumonia_severe":
        # Neumonía severa
        clinical_data = np.array([[0.7, 1, 1, 1, 0.94, 0.75, 0.65, 0.89]])
    else:
        # Caso moderado por defecto
        clinical_data = np.array([[0.6, 1, 1, 1, 0.92, 0.7, 0.6, 0.92]])
    
    # Realizar predicción
    predictions = model.predict([img_array, clinical_data], verbose=0)
    
    # Interpretar resultados
    detection_prob = predictions[0][0][0]
    type_probs = predictions[1][0]
    severity_probs = predictions[2][0]
    triage_score = predictions[3][0][0]
    
    # Mostrar resultados
    print(f" RESULTADOS:")
    print(f"    Detección: {'Neumonía' if detection_prob > 0.5 else 'Normal'} ({detection_prob:.3f})")
    print(f"    Tipo: Viral({type_probs[0]:.3f}) | Bacterial({type_probs[1]:.3f}) | Atípica({type_probs[2]:.3f})")
    print(f"    Severidad: Leve({severity_probs[0]:.3f}) | Moderada({severity_probs[1]:.3f}) | Severa({severity_probs[2]:.3f})")
    print(f"    Triage: {triage_score:.1f}/10")
    
    return {
        'image_path': image_path,
        'detection': detection_prob,
        'type': type_probs,
        'severity': severity_probs,
        'triage': triage_score
    }

def test_multiple_images():
    """Probar con múltiples imágenes del dataset"""
    
    # Cargar modelo
    model = load_model()
    
    # Definir rutas de imágenes de test
    test_cases = [
        ("data/chest_xray/test/NORMAL", "normal"),
        ("data/chest_xray/test/PNEUMONIA", "pneumonia_moderate")
    ]
    
    results = []
    
    for folder, case_type in test_cases:
        if os.path.exists(folder):
            # Tomar las primeras 3 imágenes de cada carpeta
            image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for i, img_file in enumerate(image_files[:3]):
                img_path = os.path.join(folder, img_file)
                result = test_with_real_image(model, img_path, case_type)
                if result:
                    results.append(result)
        else:
            print(f" Carpeta no encontrada: {folder}")
    
    return results

if __name__ == "__main__":
    print(" PneumoScan AI - Test con Imágenes Reales")
    print("=" * 50)
    
    results = test_multiple_images()
    
    if results:
        print(f"\n RESUMEN DE {len(results)} CASOS ANALIZADOS:")
        print("=" * 50)
        
        for i, result in enumerate(results, 1):
            diagnosis = "Neumonía" if result['detection'] > 0.5 else "Normal"
            triage_level = "🔴" if result['triage'] >= 8 else "🟠" if result['triage'] >= 6 else "🟡" if result['triage'] >= 3 else "🟢"
            
            print(f"Caso {i}: {diagnosis} | Triage: {result['triage']:.1f} {triage_level}")
    
    print("\n Test completado!")