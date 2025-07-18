import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from datetime import datetime

class PneumoScanDiagnostic:
    """Clase principal para diagnóstico médico"""
    
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Cargar modelo entrenado"""
        print(" Inicializando PneumoScan AI...")
        
        custom_objects = {
            'mse': tf.keras.losses.MeanSquaredError(),
            'mae': tf.keras.metrics.MeanAbsoluteError(),
            'binary_crossentropy': tf.keras.losses.BinaryCrossentropy(),
            'sparse_categorical_crossentropy': tf.keras.losses.SparseCategoricalCrossentropy(),
            'accuracy': tf.keras.metrics.Accuracy()
        }
        
        self.model = tf.keras.models.load_model(
            'models/cpu_local/best_model_cpu.h5',
            custom_objects=custom_objects,
            compile=False
        )
        
        print(" Sistema listo para diagnóstico!")
    
    def diagnose_pneumonia(self, image_path, patient_data):
        """
        Diagnóstico completo de neumonía
        
        Args:
            image_path: Ruta a la radiografía
            patient_data: Dict con datos del paciente
        """
        
        # Procesar imagen
        if not os.path.exists(image_path):
            return {"error": f"Imagen no encontrada: {image_path}"}
        
        img = image.load_img(image_path, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Procesar datos clínicos
        clinical_features = self._process_clinical_data(patient_data)
        clinical_array = np.expand_dims(clinical_features, axis=0)
        
        # Realizar predicción
        predictions = self.model.predict([img_array, clinical_array], verbose=0)
        
        # Interpretar resultados
        return self._interpret_results(predictions, patient_data)
    
    def _process_clinical_data(self, patient_data):
        """Procesar datos clínicos del paciente"""
        
        # Valores por defecto
        defaults = {
            'age': 50, 'fever': 0, 'cough': 0, 'dyspnea': 0,
            'temperature': 37.0, 'heart_rate': 80, 'respiratory_rate': 16, 'spo2': 98
        }
        
        # Aplicar valores del paciente
        for key in defaults:
            if key in patient_data:
                defaults[key] = patient_data[key]
        
        # Normalizar características
        features = np.array([
            defaults['age'] / 100.0,           # age_normalized
            1 if defaults['fever'] else 0,     # fever
            1 if defaults['cough'] else 0,     # cough
            1 if defaults['dyspnea'] else 0,   # dyspnea
            defaults['temperature'] / 42.0,    # temperature
            defaults['heart_rate'] / 150.0,    # heart_rate
            defaults['respiratory_rate'] / 40.0, # respiratory_rate
            defaults['spo2'] / 100.0           # spo2
        ], dtype=np.float32)
        
        return features
    
    def _interpret_results(self, predictions, patient_data):
        """Interpretar resultados de predicción"""
        
        detection_prob = predictions[0][0][0]
        type_probs = predictions[1][0]
        severity_probs = predictions[2][0]
        triage_score = predictions[3][0][0]
        
        # Determinar diagnóstico principal
        has_pneumonia = detection_prob > 0.5
        
        # Determinar tipo más probable
        type_names = ['Viral', 'Bacterial', 'Atípica']
        type_index = np.argmax(type_probs)
        pneumonia_type = type_names[type_index]
        type_confidence = type_probs[type_index]
        
        # Determinar severidad
        severity_names = ['Leve', 'Moderada', 'Severa']
        severity_index = np.argmax(severity_probs)
        severity_level = severity_names[severity_index]
        severity_confidence = severity_probs[severity_index]
        
        # Determinar prioridad de triage
        if triage_score >= 8.0:
            triage_priority = "CRÍTICA"
            triage_color = "🔴"
            triage_action = "Atención inmediata"
        elif triage_score >= 6.0:
            triage_priority = "URGENTE"
            triage_color = "🟠"
            triage_action = "Atención en 30 minutos"
        elif triage_score >= 3.0:
            triage_priority = "MODERADA"
            triage_color = "🟡"
            triage_action = "Atención en 2 horas"
        else:
            triage_priority = "BAJA"
            triage_color = "🟢"
            triage_action = "Atención programada"
        
        return {
            'timestamp': datetime.now().isoformat(),
            'patient_data': patient_data,
            'diagnosis': {
                'has_pneumonia': has_pneumonia,
                'confidence': float(detection_prob),
                'primary_diagnosis': 'Neumonía detectada' if has_pneumonia else 'Radiografía normal'
            },
            'pneumonia_details': {
                'type': pneumonia_type,
                'type_confidence': float(type_confidence),
                'severity': severity_level,
                'severity_confidence': float(severity_confidence)
            },
            'triage': {
                'score': float(triage_score),
                'priority': triage_priority,
                'color': triage_color,
                'recommended_action': triage_action
            },
            'raw_predictions': {
                'detection': float(detection_prob),
                'type_probabilities': type_probs.tolist(),
                'severity_probabilities': severity_probs.tolist(),
                'triage_score': float(triage_score)
            }
        }
    
    def generate_medical_report(self, diagnosis_result):
        """Generar reporte médico completo"""
        
        patient = diagnosis_result['patient_data']
        diagnosis = diagnosis_result['diagnosis']
        details = diagnosis_result['pneumonia_details']
        triage = diagnosis_result['triage']
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                     REPORTE PNEUMOSCAN AI                    ║
╚══════════════════════════════════════════════════════════════╝

INFORMACIÓN DEL PACIENTE:
─────────────────────────
• Edad: {patient.get('age', 'No especificada')} años
• Síntomas: 
  - Fiebre: {'Sí' if patient.get('fever', False) else 'No'}
  - Tos: {'Sí' if patient.get('cough', False) else 'No'}
  - Disnea: {'Sí' if patient.get('dyspnea', False) else 'No'}

SIGNOS VITALES:
─────────────────
• Temperatura: {patient.get('temperature', 'N/A')}°C
• Frecuencia cardíaca: {patient.get('heart_rate', 'N/A')} lpm
• Frecuencia respiratoria: {patient.get('respiratory_rate', 'N/A')} rpm
• SpO2: {patient.get('spo2', 'N/A')}%

DIAGNÓSTICO POR IA:
─────────────────────
 RESULTADO PRINCIPAL:
   {diagnosis['primary_diagnosis']}
   (Confianza: {diagnosis['confidence']:.1%})

{' CARACTERÍSTICAS DE LA NEUMONÍA:' if diagnosis['has_pneumonia'] else ''}
{'   • Tipo más probable: ' + details['type'] + f" ({details['type_confidence']:.1%})" if diagnosis['has_pneumonia'] else ''}
{'   • Severidad estimada: ' + details['severity'] + f" ({details['severity_confidence']:.1%})" if diagnosis['has_pneumonia'] else ''}

 EVALUACIÓN DE TRIAGE:
   • Score: {triage['score']:.1f}/10.0
   • Prioridad: {triage['color']} {triage['priority']}
   • Acción recomendada: {triage['recommended_action']}

RECOMENDACIONES CLÍNICAS:
──────────────────────────
{'• Iniciar protocolo de neumonía según tipo identificado' if diagnosis['has_pneumonia'] else '• Continuar con evaluación clínica de rutina'}
{'• Monitoreo de signos vitales cada 4 horas' if triage['score'] >= 6 else '• Seguimiento ambulatorio estándar'}
{'• Considerar hospitalización inmediata' if triage['score'] >= 8 else ''}
• Correlacionar con evaluación clínica directa
• Este diagnóstico de IA es de apoyo y no reemplaza el criterio médico

─────────────────────────────────────────────────────────────
Generado por PneumoScan AI v1.0 | {diagnosis_result['timestamp']}
IMPORTANTE: Este es un sistema de apoyo al diagnóstico.
La decisión clínica final debe basarse en evaluación médica completa.
        """
        
        return report

def interactive_diagnosis_improved():
    """Diagnóstico interactivo mejorado con verificación de rutas"""
    
    print(" PneumoScan AI - Diagnóstico Interactivo Mejorado")
    print("=" * 55)
    
    # Mostrar imágenes disponibles
    print("\n📂 Verificando imágenes disponibles...")
    base_path = Path("data/chest_xray/test")
    
    available_images = []
    for class_folder in ['NORMAL', 'PNEUMONIA']:
        folder_path = base_path / class_folder
        if folder_path.exists():
            images = list(folder_path.glob('*.jpeg'))[:5]  # Primeras 5
            available_images.extend(images)
            print(f"   {class_folder}: {len(list(folder_path.glob('*.jpeg')))} imágenes disponibles")
    
    if available_images:
        print(f"\n Ejemplos de rutas válidas:")
        for i, img in enumerate(available_images[:5], 1):
            print(f"   {i}. {img}")
    
    # Inicializar sistema
    pneumoscan = PneumoScanDiagnostic()
    
    while True:
        print("\n" + "="*60)
        print("📋 NUEVO PACIENTE")
        print("="*60)
        
        try:
            # Opción de usar imagen de ejemplo o ruta personalizada
            print("\nOpciones:")
            print("1. Usar imagen de ejemplo")
            print("2. Especificar ruta personalizada")
            print("3. Salir")
            
            choice = input("Elige opción (1/2/3): ").strip()
            
            if choice == "3":
                break
            elif choice == "1" and available_images:
                print("\nImágenes de ejemplo disponibles:")
                for i, img in enumerate(available_images[:5], 1):
                    class_name = "NORMAL" if "NORMAL" in str(img) else "PNEUMONIA"
                    print(f"   {i}. {img.name} ({class_name})")
                
                img_choice = input("Elige imagen (1-5): ").strip()
                try:
                    img_idx = int(img_choice) - 1
                    if 0 <= img_idx < len(available_images[:5]):
                        image_path = str(available_images[img_idx])
                    else:
                        print(" Opción inválida")
                        continue
                except ValueError:
                    print(" Opción inválida")
                    continue
            elif choice == "2":
                image_path = input(" Ruta de la radiografía: ").strip()
                if not image_path:
                    continue
            else:
                print(" Opción inválida")
                continue
            
            # Verificar que la imagen existe
            if not os.path.exists(image_path):
                print(f" Imagen no encontrada: {image_path}")
                print(" Usa la opción 1 para ver imágenes disponibles")
                continue
            
            print(f" Imagen encontrada: {Path(image_path).name}")
            
            # Solicitar datos del paciente
            print("\n DATOS DEL PACIENTE:")
            age = int(input("   Edad: ") or "50")
            fever = input("   ¿Tiene fiebre? (s/n): ").lower().startswith('s')
            cough = input("   ¿Tiene tos? (s/n): ").lower().startswith('s')
            dyspnea = input("   ¿Tiene disnea? (s/n): ").lower().startswith('s')
            temperature = float(input("   Temperatura (°C): ") or "37.0")
            heart_rate = int(input("   Frecuencia cardíaca (lpm): ") or "80")
            respiratory_rate = int(input("   Frecuencia respiratoria (rpm): ") or "16")
            spo2 = int(input("   SpO2 (%): ") or "98")
            
            patient_data = {
                'age': age, 'fever': fever, 'cough': cough, 'dyspnea': dyspnea,
                'temperature': temperature, 'heart_rate': heart_rate, 
                'respiratory_rate': respiratory_rate, 'spo2': spo2
            }
            
            # Realizar diagnóstico
            print("\n Analizando radiografía...")
            result = pneumoscan.diagnose_pneumonia(image_path, patient_data)
            
            if 'error' in result:
                print(f" {result['error']}")
                continue
            
            # Mostrar reporte
            report = pneumoscan.generate_medical_report(result)
            print("\n" + "="*70)
            print(report)
            print("="*70)
            
            # Guardar reporte
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/cpu_local/reports/reporte_medico_{timestamp}.txt"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(f"\n Reporte guardado: {filename}")
            
        except KeyboardInterrupt:
            print("\n Sesión interrumpida")
            break
        except ValueError as e:
            print(f" Error en los datos ingresados: {str(e)}")
            continue
        except Exception as e:
            print(f" Error inesperado: {str(e)}")
            continue
        
        # Continuar con otro paciente
        if input("\n¿Analizar otro paciente? (Enter=Sí, 'n'=No): ").strip().lower() == 'n':
            break
    
    print("\n Sesión de diagnóstico finalizada")

if __name__ == "__main__":
    interactive_diagnosis()