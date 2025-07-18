import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
import os
from pathlib import Path
import json
from datetime import datetime

def batch_analysis_fixed(folder_path, output_file="batch_results.csv"):
    
    print(" Iniciando análisis por lotes...")
    
    # Cargar modelo
    print(" Cargando modelo...")
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
    
    # Buscar imágenes
    folder = Path(folder_path)
    if not folder.exists():
        print(f" Carpeta no encontrada: {folder_path}")
        return
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(folder.glob(ext))
    
    if not image_files:
        print(f" No se encontraron imágenes en: {folder_path}")
        return
    
    print(f" Analizando {len(image_files)} imágenes...")
    
    results = []
    
    for i, img_path in enumerate(image_files):
        print(f"🔍 Procesando {i+1}/{len(image_files)}: {img_path.name}")
        
        try:
            # Cargar imagen
            img = image.load_img(img_path, target_size=(128, 128))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Datos clínicos por defecto
            clinical_data = np.array([[0.5, 0, 1, 0, 0.9, 0.6, 0.5, 0.95]])
            
            # Predicción
            predictions = model.predict([img_array, clinical_data], verbose=0)
            
            # Procesar resultados - CONVERTIR A TIPOS PYTHON NATIVOS
            detection_prob = float(predictions[0][0][0])
            type_probs = [float(x) for x in predictions[1][0]]
            severity_probs = [float(x) for x in predictions[2][0]]
            triage_score = float(predictions[3][0][0])
            
            result = {
                'filename': str(img_path.name),
                'filepath': str(img_path),
                'detection_probability': detection_prob,
                'has_pneumonia': bool(detection_prob > 0.5),  # Convertir a bool nativo
                'type_viral': type_probs[0],
                'type_bacterial': type_probs[1],
                'type_atypical': type_probs[2],
                'predicted_type': ['Viral', 'Bacterial', 'Atípica'][int(np.argmax(type_probs))],
                'severity_mild': severity_probs[0],
                'severity_moderate': severity_probs[1],
                'severity_severe': severity_probs[2],
                'predicted_severity': ['Leve', 'Moderada', 'Severa'][int(np.argmax(severity_probs))],
                'triage_score': triage_score,
                'triage_priority': 'CRÍTICA' if triage_score >= 8 else 'URGENTE' if triage_score >= 6 else 'MODERADA' if triage_score >= 3 else 'BAJA',
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            results.append(result)
            
        except Exception as e:
            print(f" Error procesando {img_path.name}: {str(e)}")
            continue
    
    # Guardar resultados
    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        
        # Crear directorio si no existe
        output_dir = Path("results/cpu_local/reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar CSV
        csv_path = output_dir / output_file
        df.to_csv(csv_path, index=False)
        
        # Guardar JSON (ahora funcionará)
        json_path = csv_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Estadísticas
        pneumonia_cases = sum(1 for r in results if r['has_pneumonia'])
        total_cases = len(results)
        avg_triage = sum(r['triage_score'] for r in results) / len(results)
        
        print(f"\n RESULTADOS DEL ANÁLISIS:")
        print(f"   • Total de imágenes: {total_cases}")
        print(f"   • Casos de neumonía detectados: {pneumonia_cases} ({pneumonia_cases/total_cases:.1%})")
        print(f"   • Score promedio de triage: {avg_triage:.1f}")
        print(f"   • Casos críticos (triage ≥8): {sum(1 for r in results if r['triage_score'] >= 8)}")
        print(f"   • Casos urgentes (triage 6-8): {sum(1 for r in results if 6 <= r['triage_score'] < 8)}")
        
        print(f"\n Resultados guardados:")
        print(f"   • CSV: {csv_path}")
        print(f"   • JSON: {json_path}")
        
        return df
    
    else:
        print(" No se pudieron procesar imágenes")
        return None