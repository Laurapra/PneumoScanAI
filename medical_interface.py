import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Any
import json
import random
from dataclasses import dataclass, asdict


#configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PneumoScanConfig:
    """Configuración centralizada para PneumoScan AI"""
    
    #rutas del sistema
    MODEL_PATH = 'models/cpu_local/best_model_cpu.h5'
    BASE_DATA_PATH = Path("data/chest_xray/test")
    REPORTS_PATH = Path("results/cpu_local/reports")
    TEST_DATA_PATH = Path("test_data")
    
    #parámetros del modelo
    IMAGE_SIZE = (128, 128)
    
    #valores por defecto para datos clínicos
    DEFAULT_CLINICAL_VALUES = {
        'age': 50,
        'fever': False,
        'cough': False,
        'dyspnea': False,
        'temperature': 37.0,
        'heart_rate': 80,
        'respiratory_rate': 16,
        'spo2': 98
    }
    
    #umbrales para clasificación
    PNEUMONIA_THRESHOLD = 0.5
    TRIAGE_THRESHOLDS = {
        'critical': 8.0,
        'urgent': 6.0,
        'moderate': 3.0
    }
    
    #etiquetas para clasificación
    TYPE_LABELS = ['Viral', 'Bacterial', 'Atípica']
    SEVERITY_LABELS = ['Leve', 'Moderada', 'Severa']
    CLASS_FOLDERS = ['NORMAL', 'PNEUMONIA']


@dataclass
class TestPatient:
    """Estructura para pacientes de prueba"""
    id: str
    name: str
    age: int
    fever: bool
    cough: bool
    dyspnea: bool
    temperature: float
    heart_rate: int
    respiratory_rate: int
    spo2: int
    expected_diagnosis: str
    case_description: str
    image_type: str  # 'normal' o 'pneumonia'


class TestDataGenerator:
    """Generador de datos de prueba para el sistema"""
    
    def __init__(self):
        self.test_patients = self._generate_test_patients()
        self._setup_test_environment()
    
    def _generate_test_patients(self) -> List[TestPatient]:
        """Genera pacientes de prueba con casos variados"""
        patients = [
            #casos normales
            TestPatient(
                id="P001",
                name="María González",
                age=35,
                fever=False,
                cough=False,
                dyspnea=False,
                temperature=36.8,
                heart_rate=72,
                respiratory_rate=16,
                spo2=98,
                expected_diagnosis="Normal",
                case_description="Paciente sana, examen de rutina",
                image_type="normal"
            ),
            TestPatient(
                id="P002",
                name="Carlos Rodríguez",
                age=42,
                fever=False,
                cough=True,
                dyspnea=False,
                temperature=37.1,
                heart_rate=68,
                respiratory_rate=14,
                spo2=97,
                expected_diagnosis="Normal",
                case_description="Tos leve sin otros síntomas",
                image_type="normal"
            ),
            
            #casos de neumonía leve
            TestPatient(
                id="P003",
                name="Ana Martínez",
                age=28,
                fever=True,
                cough=True,
                dyspnea=False,
                temperature=38.2,
                heart_rate=88,
                respiratory_rate=18,
                spo2=96,
                expected_diagnosis="Neumonía Leve",
                case_description="Neumonía adquirida en comunidad, síntomas leves",
                image_type="pneumonia"
            ),
            TestPatient(
                id="P004",
                name="Pedro Sánchez",
                age=45,
                fever=True,
                cough=True,
                dyspnea=True,
                temperature=38.5,
                heart_rate=92,
                respiratory_rate=20,
                spo2=94,
                expected_diagnosis="Neumonía Moderada",
                case_description="Neumonía con síntomas moderados",
                image_type="pneumonia"
            ),
            
            #casos severos
            TestPatient(
                id="P005",
                name="Rosa López",
                age=67,
                fever=True,
                cough=True,
                dyspnea=True,
                temperature=39.2,
                heart_rate=110,
                respiratory_rate=28,
                spo2=88,
                expected_diagnosis="Neumonía Severa",
                case_description="Paciente mayor con neumonía severa, requiere hospitalización",
                image_type="pneumonia"
            ),
            TestPatient(
                id="P006",
                name="Eduardo Vargas",
                age=78,
                fever=True,
                cough=True,
                dyspnea=True,
                temperature=39.8,
                heart_rate=125,
                respiratory_rate=32,
                spo2=85,
                expected_diagnosis="Neumonía Crítica",
                case_description="Paciente crítico con insuficiencia respiratoria",
                image_type="pneumonia"
            ),
            
            #casos pediátricos
            TestPatient(
                id="P007",
                name="Sofía Jiménez",
                age=8,
                fever=True,
                cough=True,
                dyspnea=False,
                temperature=38.0,
                heart_rate=100,
                respiratory_rate=24,
                spo2=95,
                expected_diagnosis="Neumonía Pediátrica",
                case_description="Caso pediátrico con neumonía leve",
                image_type="pneumonia"
            ),
            
            #casos geriátricos
            TestPatient(
                id="P008",
                name="Manuel Torres",
                age=85,
                fever=False,  #the elderly may not have a frver
                cough=True,
                dyspnea=True,
                temperature=36.5,
                heart_rate=95,
                respiratory_rate=22,
                spo2=91,
                expected_diagnosis="Neumonía Atípica",
                case_description="Presentación atípica en paciente geriátrico",
                image_type="pneumonia"
            ),
            
            #casos con comorbilidades
            TestPatient(
                id="P009",
                name="Carmen Ruiz",
                age=55,
                fever=True,
                cough=True,
                dyspnea=True,
                temperature=38.7,
                heart_rate=105,
                respiratory_rate=25,
                spo2=90,
                expected_diagnosis="Neumonía con Comorbilidades",
                case_description="Paciente diabética con neumonía",
                image_type="pneumonia"
            ),
            
            #caso borderline
            TestPatient(
                id="P010",
                name="Luis Herrera",
                age=39,
                fever=True,
                cough=True,
                dyspnea=False,
                temperature=37.8,
                heart_rate=85,
                respiratory_rate=19,
                spo2=96,
                expected_diagnosis="Sospecha de Neumonía",
                case_description="Caso límite que requiere evaluación adicional",
                image_type="pneumonia"
            )
        ]
        
        return patients
    
    def _setup_test_environment(self) -> None:
        """Configura el entorno de pruebas"""
        #crear directorio de datos de prueba
        PneumoScanConfig.TEST_DATA_PATH.mkdir(exist_ok=True)
        
        #guardar pacientes de prueba en JSON
        patients_file = PneumoScanConfig.TEST_DATA_PATH / "test_patients.json"
        with open(patients_file, 'w', encoding='utf-8') as f:
            patients_data = [asdict(patient) for patient in self.test_patients]
            json.dump(patients_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f" Datos de prueba guardados en: {patients_file}")
    
    def get_test_patients(self) -> List[TestPatient]:
        """Retorna la lista de pacientes de prueba"""
        return self.test_patients
    
    def get_patient_by_id(self, patient_id: str) -> Optional[TestPatient]:
        """Obtiene un paciente específico por ID"""
        for patient in self.test_patients:
            if patient.id == patient_id:
                return patient
        return None
    
    def generate_synthetic_image_path(self, image_type: str) -> str:
        """Genera una ruta de imagen sintética para pruebas"""
        if image_type == "normal":
            return f"test_data/synthetic_normal_{random.randint(1,100)}.jpg"
        else:
            return f"test_data/synthetic_pneumonia_{random.randint(1,100)}.jpg"


class MockImageProcessor:
    """Procesador de imágenes mock para pruebas sin imágenes reales"""
    
    @staticmethod
    def validate_image_path(image_path: str) -> bool:
        """Mock que siempre valida como True para pruebas"""
        if "synthetic" in image_path or "test_data" in image_path:
            return True
        
        # Validación real para imágenes existentes
        path = Path(image_path)
        if not path.exists():
            logger.warning(f"Imagen no encontrada: {image_path}")
            return False
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        if path.suffix.lower() not in valid_extensions:
            logger.error(f"Formato de imagen no soportado: {path.suffix}")
            return False
        
        return True
    
    @staticmethod
    def preprocess_image(image_path: str) -> Optional[np.ndarray]:
        """Preprocesa imagen o genera datos sintéticos para pruebas"""
        try:
            if not MockImageProcessor.validate_image_path(image_path):
                return None
            
            # Si es una imagen sintética, generar datos mock
            if "synthetic" in image_path:
                # Generar imagen sintética basada en el tipo
                if "normal" in image_path:
                    # Imagen "normal" - valores más uniformes
                    img_array = np.random.uniform(0.3, 0.7, (1, 128, 128, 3)).astype(np.float32)
                else:
                    # Imagen "pneumonia" - más variabilidad
                    img_array = np.random.uniform(0.1, 0.9, (1, 128, 128, 3)).astype(np.float32)
                
                logger.info(f"🔬 Imagen sintética generada para: {Path(image_path).name}")
                return img_array
            
            # Procesamiento real para imágenes existentes
            img = image.load_img(image_path, target_size=PneumoScanConfig.IMAGE_SIZE)
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            logger.error(f"Error al procesar imagen: {e}")
            return None


class PatientData:
    """Clase para manejar datos del paciente de forma estructurada"""
    
    def __init__(self, age: int = 50, fever: bool = False, cough: bool = False,
                 dyspnea: bool = False, temperature: float = 37.0,
                 heart_rate: int = 80, respiratory_rate: int = 16, spo2: int = 98):
        self.age = age
        self.fever = fever
        self.cough = cough
        self.dyspnea = dyspnea
        self.temperature = temperature
        self.heart_rate = heart_rate
        self.respiratory_rate = respiratory_rate
        self.spo2 = spo2
    
    @classmethod
    def from_test_patient(cls, test_patient: TestPatient) -> 'PatientData':
        """Crea PatientData desde un TestPatient"""
        return cls(
            age=test_patient.age,
            fever=test_patient.fever,
            cough=test_patient.cough,
            dyspnea=test_patient.dyspnea,
            temperature=test_patient.temperature,
            heart_rate=test_patient.heart_rate,
            respiratory_rate=test_patient.respiratory_rate,
            spo2=test_patient.spo2
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte los datos del paciente a diccionario"""
        return {
            'age': self.age,
            'fever': self.fever,
            'cough': self.cough,
            'dyspnea': self.dyspnea,
            'temperature': self.temperature,
            'heart_rate': self.heart_rate,
            'respiratory_rate': self.respiratory_rate,
            'spo2': self.spo2
        }
    
    def validate(self) -> bool:
        """Valida que los datos del paciente estén en rangos aceptables"""
        try:
            assert 0 <= self.age <= 120, "Edad debe estar entre 0 y 120 años"
            assert 30.0 <= self.temperature <= 45.0, "Temperatura debe estar entre 30°C y 45°C"
            assert 30 <= self.heart_rate <= 250, "Frecuencia cardíaca debe estar entre 30 y 250 lpm"
            assert 5 <= self.respiratory_rate <= 60, "Frecuencia respiratoria debe estar entre 5 y 60 rpm"
            assert 50 <= self.spo2 <= 100, "SpO2 debe estar entre 50% y 100%"
            return True
        except AssertionError as e:
            logger.error(f"Error de validación: {e}")
            return False


class DiagnosisResult:
    """Clase para estructurar los resultados del diagnóstico"""
    
    def __init__(self, patient_data: PatientData, predictions: List[np.ndarray], test_patient: Optional[TestPatient] = None):
        self.timestamp = datetime.now().isoformat()
        self.patient_data = patient_data
        self.test_patient = test_patient
        self._process_predictions(predictions)
    
    def _process_predictions(self, predictions: List[np.ndarray]) -> None:
        """Procesa las predicciones del modelo"""
        detection_prob = predictions[0][0][0]
        type_probs = predictions[1][0]
        severity_probs = predictions[2][0]
        triage_score = predictions[3][0][0]
        
        #diagnóstico principal
        self.has_pneumonia = detection_prob > PneumoScanConfig.PNEUMONIA_THRESHOLD
        self.detection_confidence = float(detection_prob)
        
        #tipo de neumonía
        type_index = np.argmax(type_probs)
        self.pneumonia_type = PneumoScanConfig.TYPE_LABELS[type_index]
        self.type_confidence = float(type_probs[type_index])
        
        #severidad
        severity_index = np.argmax(severity_probs)
        self.severity_level = PneumoScanConfig.SEVERITY_LABELS[severity_index]
        self.severity_confidence = float(severity_probs[severity_index])
        
        #triage
        self.triage_score = float(triage_score)
        self._calculate_triage_priority()
        
        #predicciones raw para análisis avanzado
        self.raw_predictions = {
            'detection': float(detection_prob),
            'type_probabilities': type_probs.tolist(),
            'severity_probabilities': severity_probs.tolist(),
            'triage_score': float(triage_score)
        }
    
    def _calculate_triage_priority(self) -> None:
        """Calcula la prioridad de triage basada en el score"""
        thresholds = PneumoScanConfig.TRIAGE_THRESHOLDS
        
        if self.triage_score >= thresholds['critical']:
            self.triage_priority = "CRÍTICA"
            self.triage_color = "🔴"
            self.recommended_action = "Atención inmediata"
        elif self.triage_score >= thresholds['urgent']:
            self.triage_priority = "URGENTE"
            self.triage_color = "🟠"
            self.recommended_action = "Atención en 30 minutos"
        elif self.triage_score >= thresholds['moderate']:
            self.triage_priority = "MODERADA"
            self.triage_color = "🟡"
            self.recommended_action = "Atención en 2 horas"
        else:
            self.triage_priority = "BAJA"
            self.triage_color = "🟢"
            self.recommended_action = "Atención programada"


class ClinicalDataProcessor:
    """Clase para procesamiento de datos clínicos"""
    
    @staticmethod
    def normalize_clinical_features(patient_data: PatientData) -> np.ndarray:
        """Normaliza las características clínicas para el modelo"""
        features = np.array([
            patient_data.age / 100.0,
            1.0 if patient_data.fever else 0.0,
            1.0 if patient_data.cough else 0.0,
            1.0 if patient_data.dyspnea else 0.0,
            patient_data.temperature / 42.0,
            patient_data.heart_rate / 150.0,
            patient_data.respiratory_rate / 40.0,
            patient_data.spo2 / 100.0
        ], dtype=np.float32)
        
        return features


class MockModel:
    """Modelo mock para pruebas cuando no existe el modelo real"""
    
    def predict(self, inputs: List[np.ndarray], verbose: int = 0) -> List[np.ndarray]:
        """Genera predicciones sintéticas basadas en los datos clínicos"""
        img_array, clinical_array = inputs
        clinical_features = clinical_array[0]
        
        #extraer características clínicas normalizadas
        age_norm = clinical_features[0]
        fever = clinical_features[1]
        cough = clinical_features[2]
        dyspnea = clinical_features[3]
        temp_norm = clinical_features[4]
        hr_norm = clinical_features[5]
        rr_norm = clinical_features[6]
        spo2_norm = clinical_features[7]
        
        #calcular score de riesgo basado en síntomas
        risk_score = (
            fever * 0.3 +
            cough * 0.2 +
            dyspnea * 0.3 +
            max(0, temp_norm - 0.88) * 2.0 +  #> 37°C
            max(0, hr_norm - 0.6) * 1.5 +    #> 90 bpm
            max(0, rr_norm - 0.5) * 1.5 +    #> 20 rpm
            max(0, 0.95 - spo2_norm) * 3.0 +  #< 95%
            age_norm * 0.5                  #age factor
        )
        
        #detección de neumonía
        detection_prob = min(0.95, max(0.05, risk_score))
        
        #Tipo de neumonía 
        if risk_score > 0.7:
            type_probs = np.array([0.2, 0.6, 0.2])  #most likely bacterial
        elif risk_score > 0.4:
            type_probs = np.array([0.5, 0.3, 0.2])  #more likely to go viral
        else:
            type_probs = np.array([0.4, 0.3, 0.3])  #uniform distribution
        
        #severidad basada en signos vitales
        severity_score = (
            dyspnea * 0.4 +
            max(0, temp_norm - 0.9) * 2.0 +
            max(0, hr_norm - 0.7) * 1.5 +
            max(0, 0.9 - spo2_norm) * 3.0 +
            age_norm * 0.3
        )
        
        if severity_score > 0.8:
            severity_probs = np.array([0.1, 0.2, 0.7])  #severe
        elif severity_score > 0.4:
            severity_probs = np.array([0.2, 0.6, 0.2])  #moderate
        else:
            severity_probs = np.array([0.7, 0.2, 0.1])  #mild
        
        #score de triage (0-10)
        triage_score = min(10.0, risk_score * 10.0)
        
        #formatear como salidas del modelo real
        return [
            np.array([[detection_prob]]),              #detection
            np.array([type_probs]),                    #type
            np.array([severity_probs]),                #severity 
            np.array([[triage_score]])                 #triage
        ]


class ReportGenerator:
    """Generador de reportes médicos"""
    
    @staticmethod
    def generate_medical_report(result: DiagnosisResult) -> str:
        """Genera reporte médico completo"""
        patient = result.patient_data
        test_info = ""
        
        if result.test_patient:
            test_info = f"""
INFORMACIÓN DEL CASO DE PRUEBA:
─────────────────────────────────
• ID del Paciente: {result.test_patient.id}
• Nombre: {result.test_patient.name}
• Diagnóstico Esperado: {result.test_patient.expected_diagnosis}
• Descripción: {result.test_patient.case_description}
"""
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                   REPORTE PNEUMOSCAN AI                      ║
╚══════════════════════════════════════════════════════════════╝
{test_info}
INFORMACIÓN DEL PACIENTE:
─────────────────────────
• Edad: {patient.age} años
• Síntomas: 
  - Fiebre: {'Sí' if patient.fever else 'No'}
  - Tos: {'Sí' if patient.cough else 'No'}
  - Disnea: {'Sí' if patient.dyspnea else 'No'}

SIGNOS VITALES:
─────────────────
• Temperatura: {patient.temperature}°C
• Frecuencia cardíaca: {patient.heart_rate} lpm
• Frecuencia respiratoria: {patient.respiratory_rate} rpm
• SpO2: {patient.spo2}%

DIAGNÓSTICO POR IA:
─────────────────────
🔍 RESULTADO PRINCIPAL:
   {'Neumonía detectada' if result.has_pneumonia else 'Radiografía normal'}
   (Confianza: {result.detection_confidence:.1%})

{f'''🦠 CARACTERÍSTICAS DE LA NEUMONÍA:
   • Tipo más probable: {result.pneumonia_type} ({result.type_confidence:.1%})
   • Severidad estimada: {result.severity_level} ({result.severity_confidence:.1%})''' if result.has_pneumonia else ''}

🏥 EVALUACIÓN DE TRIAGE:
   • Score: {result.triage_score:.1f}/10.0
   • Prioridad: {result.triage_color} {result.triage_priority}
   • Acción recomendada: {result.recommended_action}

RECOMENDACIONES CLÍNICAS:
──────────────────────────
{ReportGenerator._get_clinical_recommendations(result)}

─────────────────────────────────────────────────────────────
Generado por PneumoScan AI v1.2.0 | {result.timestamp}
IMPORTANTE: Este es un sistema de apoyo al diagnóstico.
La decisión clínica final debe basarse en evaluación médica completa.
        """
        
        return report
    
    @staticmethod
    def _get_clinical_recommendations(result: DiagnosisResult) -> str:
        """Genera recomendaciones clínicas específicas"""
        recommendations = []
        
        if result.has_pneumonia:
            recommendations.append("• Iniciar protocolo de neumonía según tipo identificado")
            
            if result.triage_score >= PneumoScanConfig.TRIAGE_THRESHOLDS['critical']:
                recommendations.append("• Considerar hospitalización inmediata")
                recommendations.append("• Monitoreo de signos vitales continuo")
            elif result.triage_score >= PneumoScanConfig.TRIAGE_THRESHOLDS['urgent']:
                recommendations.append("• Monitoreo de signos vitales cada 2 horas")
                recommendations.append("• Evaluar necesidad de hospitalización")
            else:
                recommendations.append("• Monitoreo de signos vitales cada 4-6 horas")
        else:
            recommendations.append("• Continuar con evaluación clínica de rutina")
            recommendations.append("• Seguimiento ambulatorio estándar")
        
        recommendations.extend([
            "• Correlacionar con evaluación clínica directa",
            "• Este diagnóstico de IA es de apoyo y no reemplaza el criterio médico"
        ])
        
        return '\n'.join(recommendations)
    
    @staticmethod
    def save_report(report: str, patient_age: int, patient_id: str = None) -> str:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            patient_info = f"{patient_id}_" if patient_id else ""
            filename = PneumoScanConfig.REPORTS_PATH / f"reporte_{patient_info}{timestamp}_edad{patient_age}.txt"
            
            #crea el directorio si no existe
            filename.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info(f"Reporte guardado: {filename}")
            return str(filename)
        except Exception as e:
            logger.error(f"Error al guardar reporte: {e}")
            return ""


class PneumoScanDiagnostic:
    
    def __init__(self, use_mock_model: bool = False):
        self.model = None
        self.use_mock = use_mock_model
        self.image_processor = MockImageProcessor()
        self.clinical_processor = ClinicalDataProcessor()
        self.report_generator = ReportGenerator()
        self._load_model()
    
    def _load_model(self) -> None:
        try:
            logger.info(" Inicializando PneumoScan AI...")
            
            if self.use_mock or not Path(PneumoScanConfig.MODEL_PATH).exists():
                logger.warning("  Usando modelo mock para pruebas (modelo real no encontrado)")
                self.model = MockModel()
                self.use_mock = True
            else:
                custom_objects = {
                    'mse': tf.keras.losses.MeanSquaredError(),
                    'mae': tf.keras.metrics.MeanAbsoluteError(),
                    'binary_crossentropy': tf.keras.losses.BinaryCrossentropy(),
                    'sparse_categorical_crossentropy': tf.keras.losses.SparseCategoricalCrossentropy(),
                    'accuracy': tf.keras.metrics.Accuracy()
                }
                
                self.model = tf.keras.models.load_model(
                    PneumoScanConfig.MODEL_PATH,
                    custom_objects=custom_objects,
                    compile=False
                )
                logger.info(" Modelo real cargado exitosamente")
            
            logger.info(" Sistema listo para diagnóstico!")
            
        except Exception as e:
            logger.error(f"Error al cargar modelo: {e}")
            logger.info(" Cambiando a modo mock...")
            self.model = MockModel()
            self.use_mock = True
    
    def diagnose_pneumonia(self, image_path: str, patient_data: PatientData,
                           test_patient: Optional[TestPatient] = None) -> Optional[DiagnosisResult]:
        if not patient_data.validate():
            logger.error(" Datos del paciente no válidos. No se puede realizar el diagnóstico.")
            return None

        # 1. Preprocesar la imagen
        processed_image = self.image_processor.preprocess_image(image_path)
        if processed_image is None:
            logger.error(f" No se pudo procesar la imagen: {image_path}")
            return None

        # 2. Preprocesar datos clínicos
        normalized_clinical_features = self.clinical_processor.normalize_clinical_features(patient_data)
        # expandir dimensiones para que coincida con el formato de entrada esperado por el modelo (batch_size, num_features)
        normalized_clinical_features = np.expand_dims(normalized_clinical_features, axis=0)

        # 3. Realizar predicciones con el modelo
        try:
            #asumiendo que el modelo espera una lista de entradas: [imagen, datos_clinicos]
            predictions = self.model.predict([processed_image, normalized_clinical_features], verbose=0)
            logger.info(" Predicciones del modelo generadas.")
        except Exception as e:
            logger.error(f" Error al realizar predicciones con el modelo: {e}")
            return None

        # 4. Procesar resultados y generar objeto DiagnosisResult
        diagnosis_result = DiagnosisResult(patient_data, predictions, test_patient)
        logger.info(" Resultados del diagnóstico procesados.")

        # 5. Generar reporte médico
        report = self.report_generator.generate_medical_report(diagnosis_result)
        
        # 6. Guardar el reporte
        saved_report_path = self.report_generator.save_report(
            report,
            patient_data.age,
            test_patient.id if test_patient else None
        )
        if not saved_report_path:
            logger.error(" No se pudo guardar el reporte.")
            return None
        
        logger.info(f" Diagnóstico completado para el paciente. Reporte guardado en: {saved_report_path}")
        return diagnosis_result

if __name__ == "__main__":
    #asegúrate de que los directorios existan para que el mock funcione
    PneumoScanConfig.REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    PneumoScanConfig.TEST_DATA_PATH.mkdir(parents=True, exist_ok=True)

    #inicializar el generador de datos de prueba
    test_data_generator = TestDataGenerator()
    
    #inicializar el sistema de diagnóstico (usará el modelo mock si el real no existe)
    pneumoscan = PneumoScanDiagnostic(use_mock_model=True) # Puedes cambiar a False si tienes un modelo real

    print("\n--- Ejecutando pruebas con pacientes sintéticos ---")
    for patient in test_data_generator.get_test_patients():
        logger.info(f"\n✨ Procesando paciente de prueba: {patient.name} (ID: {patient.id})")
        
        #generar una ruta de imagen sintética
        synthetic_image_path = test_data_generator.generate_synthetic_image_path(patient.image_type)
        
        #crear objeto PatientData a partir del TestPatient
        patient_data_for_diagnosis = PatientData.from_test_patient(patient)
        
        #realizar el diagnóstico
        result = pneumoscan.diagnose_pneumonia(synthetic_image_path, patient_data_for_diagnosis, patient)
        
        if result:
            print(f"\nDiagnóstico para {patient.name} (ID: {patient.id}):")
            print(f"  Diagnóstico Principal: {'Neumonía' if result.has_pneumonia else 'Normal'} (Confianza: {result.detection_confidence:.2f})")
            if result.has_pneumonia:
                print(f"  Tipo de Neumonía: {result.pneumonia_type} (Confianza: {result.type_confidence:.2f})")
                print(f"  Severidad: {result.severity_level} (Confianza: {result.severity_confidence:.2f})")
            print(f"  Score de Triage: {result.triage_score:.1f} - Prioridad: {result.triage_color} {result.triage_priority}")
            print(f"  Diagnóstico Esperado: {patient.expected_diagnosis}")
        else:
            logger.error(f"🚨 Falló el diagnóstico para el paciente: {patient.name} (ID: {patient.id})")
            
    print("\n--- Fin de las pruebas ---")

    #ejemplo de diagnóstico con datos manuales (sin paciente de prueba)
    print("\n--- Ejemplo de diagnóstico manual ---")
    manual_patient_data = PatientData(
        age=60,
        fever=True,
        cough=True,
        dyspnea=True,
        temperature=38.9,
        heart_rate=100,
        respiratory_rate=22,
        spo2=92
    )
    manual_image_path = test_data_generator.generate_synthetic_image_path("pneumonia")

    print(" Procesando paciente manual...")
    manual_result = pneumoscan.diagnose_pneumonia(manual_image_path, manual_patient_data)

    if manual_result:
        print("\nDiagnóstico para paciente manual:")
        print(f"  Diagnóstico Principal: {'Neumonía' if manual_result.has_pneumonia else 'Normal'} (Confianza: {manual_result.detection_confidence:.2f})")
        if manual_result.has_pneumonia:
            print(f"  Tipo de Neumonía: {manual_result.pneumonia_type} (Confianza: {manual_result.type_confidence:.2f})")
            print(f"  Severidad: {manual_result.severity_level} (Confianza: {manual_result.severity_confidence:.2f})")
        print(f"  Score de Triage: {manual_result.triage_score:.1f} - Prioridad: {manual_result.triage_color} {manual_result.triage_priority}")
    else:
        logger.error(" Falló el diagnóstico para el paciente manual.")