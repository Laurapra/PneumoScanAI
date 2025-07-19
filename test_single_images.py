import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from pathlib import Path
from datetime import datetime

def load_pneumoscan_model():
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
    
    #re-compilar
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

def list_available_images():
    """Mostrar imágenes disponibles en el dataset"""
    print(" Explorando imágenes disponibles...")
    
    base_path = Path("data/chest_xray")
    available_images = []
    
    for split in ['test', 'val', 'train']:
        split_path = base_path / split
        if split_path.exists():
            print(f"\n {split.upper()}:")
            
            for class_name in ['NORMAL', 'PNEUMONIA']:
                class_path = split_path / class_name
                if class_path.exists():
                    images = list(class_path.glob('*.jpeg')) + list(class_path.glob('*.jpg'))
                    print(f"   {class_name}: {len(images)} imágenes")
                    
                    #guardar algunas para mostrar como ejemplos
                    if images and len(available_images) < 10:
                        for img in images[:3]:
                            available_images.append({
                                'path': str(img),
                                'name': img.name,
                                'class': class_name,
                                'split': split
                            })
    
    return available_images

def create_clinical_data(case_type="moderate_pneumonia"):
    """Crear datos clínicos realistas según el tipo de caso"""
    
    clinical_profiles = {
        "normal": {
            "age": 45, "fever": False, "cough": False, "dyspnea": False,
            "temperature": 36.8, "heart_rate": 75, "respiratory_rate": 16, "spo2": 98
        },
        "mild_pneumonia": {
            "age": 35, "fever": True, "cough": True, "dyspnea": False,
            "temperature": 37.8, "heart_rate": 85, "respiratory_rate": 18, "spo2": 96
        },
        "moderate_pneumonia": {
            "age": 55, "fever": True, "cough": True, "dyspnea": True,
            "temperature": 38.5, "heart_rate": 95, "respiratory_rate": 22, "spo2": 94
        },
        "severe_pneumonia": {
            "age": 70, "fever": True, "cough": True, "dyspnea": True,
            "temperature": 39.2, "heart_rate": 110, "respiratory_rate": 28, "spo2": 89
        }
    }
    
    profile = clinical_profiles.get(case_type, clinical_profiles["moderate_pneumonia"])
    
    #convertir a formato normalizado (8 características)
    clinical_features = np.array([
        profile["age"] / 100.0,                           # age_normalized
        1.0 if profile["fever"] else 0.0,                 # fever
        1.0 if profile["cough"] else 0.0,                 # cough
        1.0 if profile["dyspnea"] else 0.0,               # dyspnea
        profile["temperature"] / 42.0,                    # temperature normalized
        profile["heart_rate"] / 150.0,                    # heart_rate normalized
        profile["respiratory_rate"] / 40.0,               # respiratory_rate normalized
        profile["spo2"] / 100.0                           # spo2 normalized
    ], dtype=np.float32)
    
    return clinical_features, profile

def analyze_single_image(model, image_path, clinical_data, patient_info):
    
    print(f"\n ANALIZANDO IMAGEN:")
    print(f" Archivo: {Path(image_path).name}")
    print(f" Ruta: {image_path}")
    
    #erificar que la imagen existe
    if not os.path.exists(image_path):
        print(f" Imagen no encontrada: {image_path}")
        return None
    
    #cargar y procesar imagen
    try:
        img = image.load_img(image_path, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        print(" Imagen cargada y procesada correctamente")
        
        #preparar datos clínicos
        clinical_array = np.expand_dims(clinical_data, axis=0)
        
        #realizar predicción
        print(" Realizando predicción...")
        predictions = model.predict([img_array, clinical_array], verbose=0)
        
        #interpretar resultados
        return interpret_predictions(predictions, patient_info, image_path)
        
    except Exception as e:
        print(f" Error procesando imagen: {str(e)}")
        return None

def interpret_predictions(predictions, patient_info, image_path):
    
    #extraer predicciones
    detection_prob = float(predictions[0][0][0])
    type_probs = [float(x) for x in predictions[1][0]]
    severity_probs = [float(x) for x in predictions[2][0]]
    triage_score = float(predictions[3][0][0])
    
    #determinar diagnóstico
    has_pneumonia = detection_prob > 0.5
    confidence_detection = abs(detection_prob - 0.5) * 2
    
    #tipo más probable
    type_names = ['Viral', 'Bacterial', 'Atípica']
    type_index = np.argmax(type_probs)
    predicted_type = type_names[type_index]
    type_confidence = type_probs[type_index]
    
    #ceveridad más probable
    severity_names = ['Leve', 'Moderada', 'Severa']
    severity_index = np.argmax(severity_probs)
    predicted_severity = severity_names[severity_index]
    severity_confidence = severity_probs[severity_index]
    
    #categoría de triage
    if triage_score >= 8.0:
        triage_category = "🔴 CRÍTICA"
        triage_action = "Atención inmediata"
    elif triage_score >= 6.0:
        triage_category = "🟠 URGENTE"
        triage_action = "Atención en 30 minutos"
    elif triage_score >= 3.0:
        triage_category = "🟡 MODERADA"
        triage_action = "Atención en 2 horas"
    else:
        triage_category = "🟢 BAJA"
        triage_action = "Atención programada"
    
    # Mostrar resultados completos
    print("\n" + "="*70)
    print("🏥 REPORTE DE DIAGNÓSTICO PNEUMOSCAN AI")
    print("="*70)
    
    print(f"\n INFORMACIÓN DEL PACIENTE:")
    print(f"   • Edad: {patient_info['age']} años")
    print(f"   • Fiebre: {'Sí' if patient_info['fever'] else 'No'}")
    print(f"   • Tos: {'Sí' if patient_info['cough'] else 'No'}")
    print(f"   • Disnea: {'Sí' if patient_info['dyspnea'] else 'No'}")
    print(f"   • Temperatura: {patient_info['temperature']}°C")
    print(f"   • FC: {patient_info['heart_rate']} lpm")
    print(f"   • FR: {patient_info['respiratory_rate']} rpm")
    print(f"   • SpO2: {patient_info['spo2']}%")
    
    print(f"\n RESULTADO PRINCIPAL:")
    if has_pneumonia:
        print(f"    NEUMONÍA DETECTADA")
        print(f"    Probabilidad: {detection_prob:.3f} ({detection_prob*100:.1f}%)")
        print(f"    Confianza: {confidence_detection:.3f}")
    else:
        print(f"    RADIOGRAFÍA NORMAL")
        print(f"    Probabilidad de normalidad: {1-detection_prob:.3f} ({(1-detection_prob)*100:.1f}%)")
        print(f"    Confianza: {confidence_detection:.3f}")
    
    if has_pneumonia:
        print(f"\n CARACTERÍSTICAS DE LA NEUMONÍA:")
        print(f"   • Tipo más probable: {predicted_type} ({type_confidence:.3f})")
        print(f"   • Distribución de tipos:")
        print(f"     - Viral: {type_probs[0]:.3f}")
        print(f"     - Bacterial: {type_probs[1]:.3f}")
        print(f"     - Atípica: {type_probs[2]:.3f}")
        
        print(f"\n  SEVERIDAD:")
        print(f"   • Nivel estimado: {predicted_severity} ({severity_confidence:.3f})")
        print(f"   • Distribución de severidad:")
        print(f"     - Leve: {severity_probs[0]:.3f}")
        print(f"     - Moderada: {severity_probs[1]:.3f}")
        print(f"     - Severa: {severity_probs[2]:.3f}")
    
    print(f"\n🚨 EVALUACIÓN DE TRIAGE:")
    print(f"   • Score: {triage_score:.1f}/10.0")
    print(f"   • Categoría: {triage_category}")
    print(f"   • Acción recomendada: {triage_action}")
    
    print(f"\n💡 RECOMENDACIONES:")
    if has_pneumonia:
        if triage_score >= 8:
            print(f"   •  URGENTE: Hospitalización inmediata")
        elif triage_score >= 6:
            print(f"   •  Evaluación médica urgente")
        else:
            print(f"   •  Seguimiento médico ambulatorio")
        
        print(f"   •  Considerar tratamiento según tipo: {predicted_type}")
        print(f"   •  Monitoreo de signos vitales")
    else:
        print(f"   •  Continuar con cuidados de rutina")
        print(f"   •  Re-evaluar si desarrolla síntomas")
    
    print(f"\n IMPORTANTE:")
    print(f"   Este diagnóstico de IA es de apoyo solamente.")
    print(f"   La decisión clínica final debe basarse en evaluación médica completa.")
    
    print(f"\n Análisis realizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    return {
        'image_path': image_path,
        'patient_info': patient_info,
        'has_pneumonia': has_pneumonia,
        'detection_probability': detection_prob,
        'predicted_type': predicted_type,
        'type_probabilities': type_probs,
        'predicted_severity': predicted_severity,
        'severity_probabilities': severity_probs,
        'triage_score': triage_score,
        'triage_category': triage_category
    }

def interactive_single_image_test():
    """Test interactivo con una imagen específica"""
    
    print("🏥 PneumoScan AI - Test con Imagen Específica")
    print("=" * 55)
    
    # Mostrar imágenes disponibles
    available_images = list_available_images()
    
    if not available_images:
        print(" No se encontraron imágenes en el dataset")
        return
    
    print(f"\n IMÁGENES DE EJEMPLO DISPONIBLES:")
    for i, img_info in enumerate(available_images, 1):
        print(f"   {i}. {img_info['name']} ({img_info['class']}) - {img_info['split']}")
    
    #cargar modelo
    model = load_pneumoscan_model()
    
    while True:
        print(f"\n" + "="*60)
        print(" SELECCIONAR IMAGEN PARA ANALIZAR")
        print("="*60)
        
        print("\nOpciones:")
        print("1. Usar imagen de ejemplo (lista arriba)")
        print("2. Especificar ruta completa")
        print("3. Salir")
        
        choice = input("\nElige opción (1/2/3): ").strip()
        
        if choice == "3":
            break
        elif choice == "1":
            try:
                img_num = int(input(f"Número de imagen (1-{len(available_images)}): ")) - 1
                if 0 <= img_num < len(available_images):
                    image_path = available_images[img_num]['path']
                    image_class = available_images[img_num]['class']
                    print(f" Imagen seleccionada: {Path(image_path).name} ({image_class})")
                else:
                    print(" Número inválido")
                    continue
            except ValueError:
                print(" Número inválido")
                continue
        elif choice == "2":
            image_path = input("Ruta completa de la imagen: ").strip()
            if not image_path:
                continue
            image_class = "UNKNOWN"
        else:
            print(" Opción inválida")
            continue
        
        #seleccionar perfil clínico
        print(f"\n👤 PERFIL CLÍNICO DEL PACIENTE:")
        print("1. Normal (sin síntomas)")
        print("2. Neumonía leve")
        print("3. Neumonía moderada")
        print("4. Neumonía severa")
        print("5. Personalizado")
        
        profile_choice = input("Elige perfil (1-5): ").strip()
        
        profile_map = {
            "1": "normal",
            "2": "mild_pneumonia", 
            "3": "moderate_pneumonia",
            "4": "severe_pneumonia"
        }
        
        if profile_choice in profile_map:
            clinical_data, patient_info = create_clinical_data(profile_map[profile_choice])
        elif profile_choice == "5":
            #datos personalizados
            print("Ingresa datos del paciente:")
            try:
                age = int(input("Edad: ") or "50")
                fever = input("¿Fiebre? (s/n): ").lower().startswith('s')
                cough = input("¿Tos? (s/n): ").lower().startswith('s')
                dyspnea = input("¿Disnea? (s/n): ").lower().startswith('s')
                temp = float(input("Temperatura (°C): ") or "37.0")
                hr = int(input("FC (lpm): ") or "80")
                rr = int(input("FR (rpm): ") or "16")
                spo2 = int(input("SpO2 (%): ") or "98")
                
                patient_info = {
                    'age': age, 'fever': fever, 'cough': cough, 'dyspnea': dyspnea,
                    'temperature': temp, 'heart_rate': hr, 'respiratory_rate': rr, 'spo2': spo2
                }
                
                clinical_data = np.array([
                    age/100.0, 1.0 if fever else 0.0, 1.0 if cough else 0.0, 1.0 if dyspnea else 0.0,
                    temp/42.0, hr/150.0, rr/40.0, spo2/100.0
                ], dtype=np.float32)
                
            except ValueError:
                print(" Error en los datos. Usando perfil moderado por defecto.")
                clinical_data, patient_info = create_clinical_data("moderate_pneumonia")
        else:
            print("Usando perfil moderado por defecto")
            clinical_data, patient_info = create_clinical_data("moderate_pneumonia")
        
        #realizar análisis
        result = analyze_single_image(model, image_path, clinical_data, patient_info)
        
        if result:
            #guardar reporte
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/cpu_local/reports/analisis_individual_{timestamp}.txt"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            
            print(f"\n Análisis completado exitosamente")
        
        #continuar con otra imagen
        if input("\n¿Analizar otra imagen? (Enter=Sí, 'n'=No): ").strip().lower() == 'n':
            break
    
    print("\n Sesión finalizada. ¡Gracias por usar PneumoScan AI!")

if __name__ == "__main__":
    interactive_single_image_test()