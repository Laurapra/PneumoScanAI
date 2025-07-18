"""
  PneumoScan AI - Sistema completo Multi-Tarea (CORREGIDO PARA TF 2.19)
  ----------------------------------------------------------------------
  Versión corregida que resuelve el error del generador multi-modal
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import json
from pathlib import Path
import cv2
from typing import Dict, List, Tuple, Optional, Union
import logging

# Configuración optimizada para CPU
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '4'

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Importaciones de ML/DL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Importaciones de métricas
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, accuracy_score, f1_score,
    mean_absolute_error, r2_score
)

# Configurar TensorFlow
tf.random.set_seed(42)
np.random.seed(42)

print(f"TensorFlow version: {tf.__version__}")
print(f"Configurado para CPU - Optimizado para ejecución local")
print(f"Cores disponibles: {os.cpu_count()}")

class EnhancedConfig:
    """Configuración optimizada para CPU local"""

    PROJECT_NAME = "PneumoScan AI - CPU Local"
    VERSION = "1.0.0-CPU"
    TEAM = "DevSharks BQ"

    # Rutas de datos
    DATA_PATH = "data/chest_xray"
    TRAIN_PATH = f"{DATA_PATH}/train"
    VAL_PATH = f"{DATA_PATH}/val"
    TEST_PATH = f"{DATA_PATH}/test"

    # Directorios de salida
    OUTPUT_BASE = "results/cpu_local"
    MODELS_DIR = "models/cpu_local"
    PLOTS_DIR = f"{OUTPUT_BASE}/plots"
    REPORTS_DIR = f"{OUTPUT_BASE}/reports"

    # Parámetros optimizados para CPU
    IMG_SIZE = (128, 128)
    BATCH_SIZE = 4
    CHANNELS = 3
    EPOCHS = 10
    LEARNING_RATE = 0.001
    PATIENCE = 5
    MIN_LR = 1e-6

    # Configuración multi-tarea simplificada
    TASKS = {
        'detection': {
            'classes': 2,
            'type': 'binary',
            'loss': 'binary_crossentropy',
            'weight': 1.0,
            'metrics': ['accuracy']
        },
        'pneumonia_type': {
            'classes': 3,
            'type': 'categorical',
            'loss': 'sparse_categorical_crossentropy',
            'weight': 0.5,
            'metrics': ['accuracy']
        },
        'severity': {
            'classes': 3,
            'type': 'categorical',
            'loss': 'sparse_categorical_crossentropy',
            'weight': 0.5,
            'metrics': ['accuracy']
        },
        'triage': {
            'classes': 1,
            'type': 'regression',
            'loss': 'mse',
            'weight': 0.3,
            'metrics': ['mae']
        }
    }

    # Datos clínicos reducidos
    CLINICAL_FEATURES = 8
    CLINICAL_FEATURE_NAMES = [
        'age_normalized', 'fever', 'cough', 'dyspnea',
        'temperature', 'heart_rate', 'respiratory_rate', 'spo2'
    ]

    # Clases
    DETECTION_CLASSES = ['Normal', 'Pneumonia']
    TYPE_CLASSES = ['Viral', 'Bacterial', 'Atypical']
    SEVERITY_CLASSES = ['Mild', 'Moderate', 'Severe']

    @classmethod
    def create_directories(cls):
        """Crear estructura de directorios necesaria"""
        directories = [
            cls.OUTPUT_BASE,
            cls.MODELS_DIR,
            cls.PLOTS_DIR,
            cls.REPORTS_DIR,
            f"{cls.OUTPUT_BASE}/logs"
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

        logger.info(f" Estructura de directorios creada para CPU")

class ClinicalDataGenerator:
    """Generador simplificado de datos clínicos para CPU"""

    def __init__(self, seed: int = 42):
        np.random.seed(seed)

    def generate_clinical_features(self, diagnosis: str, pneumonia_type: str = None,
                                   severity: str = None) -> np.ndarray:
        """Generar 8 características clínicas simplificadas"""
        features = np.zeros(8, dtype=np.float32)

        if diagnosis == 'normal':
            features[0] = np.random.uniform(0.2, 0.6)  # age_normalized
            features[1] = np.random.choice([0, 1], p=[0.95, 0.05])  # fever
            features[2] = np.random.choice([0, 1], p=[0.9, 0.1])   # cough
            features[3] = np.random.choice([0, 1], p=[0.95, 0.05]) # dyspnea
            features[4] = np.random.uniform(0.86, 0.89)  # temperature
            features[5] = np.random.uniform(0.4, 0.6)    # heart_rate
            features[6] = np.random.uniform(0.3, 0.5)    # respiratory_rate
            features[7] = np.random.uniform(0.96, 1.0)   # spo2
        else:
            features[0] = np.random.uniform(0.4, 0.8)    # age_normalized
            features[1] = np.random.choice([0, 1], p=[0.2, 0.8])   # fever
            features[2] = np.random.choice([0, 1], p=[0.1, 0.9])   # cough
            features[3] = np.random.choice([0, 1], p=[0.4, 0.6])   # dyspnea
            features[4] = np.random.uniform(0.90, 0.95)  # temperature
            features[5] = np.random.uniform(0.6, 0.8)    # heart_rate
            features[6] = np.random.uniform(0.5, 0.7)    # respiratory_rate
            features[7] = np.random.uniform(0.88, 0.96)  # spo2

        return features

    def create_synthetic_labels(self, original_labels: np.ndarray) -> List[Dict]:
        """Crear etiquetas sintéticas simplificadas"""
        synthetic_data = []
        type_distribution = [0.3, 0.5, 0.2]

        for label in original_labels:
            if label == 0:  # Normal
                data = {
                    'detection': 0,
                    'pneumonia_type': 0,
                    'severity': 0,
                    'triage_score': np.random.uniform(1.0, 3.0),
                    'clinical_features': self.generate_clinical_features('normal')
                }
            else:  # Pneumonia
                ptype = np.random.choice([0, 1, 2], p=type_distribution)
                severity = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
                triage_score = np.random.uniform(3.0, 9.0)

                data = {
                    'detection': 1,
                    'pneumonia_type': ptype,
                    'severity': severity,
                    'triage_score': triage_score,
                    'clinical_features': self.generate_clinical_features('pneumonia', ptype, severity)
                }

            synthetic_data.append(data)

        logger.info(f" Generadas {len(synthetic_data)} etiquetas sintéticas")
        return synthetic_data

class PneumoScanMultiTaskModel:
    """Modelo simplificado para CPU"""

    def __init__(self, config: EnhancedConfig):
        self.config = config

    def create_complete_model(self) -> Model:
        """Crear modelo optimizado para CPU"""
        logger.info(" Construyendo modelo optimizado para CPU...")

        # Entradas
        image_input = layers.Input(
            shape=(*self.config.IMG_SIZE, self.config.CHANNELS),
            name='image_input'
        )
        clinical_input = layers.Input(
            shape=(self.config.CLINICAL_FEATURES,),
            name='clinical_input'
        )

        # Backbone ligero para CPU
        image_features = self._create_lightweight_backbone(image_input)
        clinical_features = self._process_clinical_simple(clinical_input)

        # Fusión simple
        fused_features = self._create_simple_fusion(image_features, clinical_features)

        # Cabezas de salida simplificadas
        outputs = self._create_output_heads(fused_features)

        model = Model(
            inputs=[image_input, clinical_input],
            outputs=list(outputs.values()),
            name='PneumoScan_CPU_Optimized'
        )

        logger.info(f" Modelo creado: {model.count_params():,} parámetros")
        return model

    def _create_lightweight_backbone(self, image_input):
        """Backbone ligero usando MobileNetV2"""
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_tensor=image_input,
            alpha=0.75
        )

        # Congelar la mayoría de capas
        for layer in base_model.layers[:-10]:
            layer.trainable = False

        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)

        return x

    def _process_clinical_simple(self, clinical_input):
        """Procesamiento simple de datos clínicos"""
        x = layers.Dense(32, activation='relu')(clinical_input)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(16, activation='relu')(x)
        return x

    def _create_simple_fusion(self, image_features, clinical_features):
        """Fusión simple"""
        fused = layers.Concatenate()([image_features, clinical_features])
        x = layers.Dense(128, activation='relu')(fused)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        return x

    def _create_output_heads(self, features):
        """Cabezas de salida simplificadas"""
        outputs = {}

        # Detección
        x1 = layers.Dense(32, activation='relu')(features)
        x1 = layers.Dropout(0.2)(x1)
        outputs['detection_output'] = layers.Dense(1, activation='sigmoid', name='detection_output')(x1)

        # Tipo
        x2 = layers.Dense(32, activation='relu')(features)
        x2 = layers.Dropout(0.2)(x2)
        outputs['type_output'] = layers.Dense(3, activation='softmax', name='type_output')(x2)

        # Severidad
        x3 = layers.Dense(32, activation='relu')(features)
        x3 = layers.Dropout(0.2)(x3)
        outputs['severity_output'] = layers.Dense(3, activation='softmax', name='severity_output')(x3)

        # Triage
        x4 = layers.Dense(32, activation='relu')(features)
        x4 = layers.Dropout(0.2)(x4)
        x4 = layers.Dense(1, activation='sigmoid')(x4)
        outputs['triage_output'] = layers.Lambda(lambda x: x * 9 + 1, name='triage_output')(x4)

        return outputs

    def compile_model(self, model: Model) -> Model:
        """Compilar modelo optimizado"""
        losses = {}
        loss_weights = {}
        metrics = {}

        for task_name, task_config in self.config.TASKS.items():
            output_name = f"{task_name.replace('pneumonia_', '')}_output"
            losses[output_name] = task_config['loss']
            loss_weights[output_name] = task_config['weight']
            metrics[output_name] = task_config['metrics']

        model.compile(
            optimizer=Adam(learning_rate=self.config.LEARNING_RATE),
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics
        )

        logger.info(" Modelo compilado para CPU")
        return model

# GENERADOR CORREGIDO PARA TENSORFLOW 2.19
class FixedMultiModalGenerator:
    """Generador corregido compatible con TensorFlow 2.19"""
    
    def __init__(self, image_generator, clinical_generator: ClinicalDataGenerator, 
                 config: EnhancedConfig):
        self.image_generator = image_generator
        self.clinical_generator = clinical_generator
        self.config = config
        
        # Generar todas las etiquetas sintéticas una vez
        self.synthetic_labels = self.clinical_generator.create_synthetic_labels(
            self.image_generator.classes
        )
        
        logger.info(f" Generador multi-modal corregido inicializado")
    
    def get_dataset(self):
        """Crear un tf.data.Dataset compatible"""
        
        def data_generator():
            """Generador interno que produce los datos"""
            for i in range(len(self.image_generator)):
                # Obtener batch de imágenes
                image_batch, _ = self.image_generator[i]
                batch_size = len(image_batch)
                
                # Preparar datos clínicos y etiquetas
                clinical_batch = np.zeros((batch_size, self.config.CLINICAL_FEATURES), dtype=np.float32)
                detection_labels = np.zeros(batch_size, dtype=np.float32)
                type_labels = np.zeros(batch_size, dtype=np.int32)
                severity_labels = np.zeros(batch_size, dtype=np.int32)
                triage_labels = np.zeros(batch_size, dtype=np.float32)
                
                # Calcular índices
                start_idx = i * self.image_generator.batch_size
                
                for j in range(batch_size):
                    if start_idx + j < len(self.synthetic_labels):
                        data = self.synthetic_labels[start_idx + j]
                        clinical_batch[j] = data['clinical_features']
                        detection_labels[j] = data['detection']
                        type_labels[j] = data['pneumonia_type']
                        severity_labels[j] = data['severity']
                        triage_labels[j] = data['triage_score']
                
                # Yield datos en formato correcto
                inputs = (image_batch.astype(np.float32), clinical_batch)
                outputs = (detection_labels, type_labels, severity_labels, triage_labels)
                
                yield inputs, outputs
        
        # Crear dataset con signatures correctas
        output_signature = (
            (
                tf.TensorSpec(shape=(None, *self.config.IMG_SIZE, self.config.CHANNELS), dtype=tf.float32),
                tf.TensorSpec(shape=(None, self.config.CLINICAL_FEATURES), dtype=tf.float32)
            ),
            (
                tf.TensorSpec(shape=(None,), dtype=tf.float32),  # detection
                tf.TensorSpec(shape=(None,), dtype=tf.int32),    # type
                tf.TensorSpec(shape=(None,), dtype=tf.int32),    # severity
                tf.TensorSpec(shape=(None,), dtype=tf.float32)   # triage
            )
        )
        
        dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=output_signature
        )
        
        return dataset
    
    def __len__(self):
        return len(self.image_generator)

class DataPipeline:
    """Pipeline optimizado para CPU con generador corregido"""

    def __init__(self, config: EnhancedConfig):
        self.config = config
        self.clinical_generator = ClinicalDataGenerator()

    def setup_data_generators(self) -> Tuple:
        """Configurar generadores optimizados para CPU"""

        # Data augmentation conservador
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=5,
            width_shift_range=0.05,
            height_shift_range=0.05,
            zoom_range=0.05,
            horizontal_flip=False,
            brightness_range=[0.95, 1.05],
            fill_mode='nearest'
        )

        val_test_datagen = ImageDataGenerator(rescale=1./255)

        # Generadores de imágenes
        train_generator = train_datagen.flow_from_directory(
            self.config.TRAIN_PATH,
            target_size=self.config.IMG_SIZE,
            batch_size=self.config.BATCH_SIZE,
            class_mode='binary',
            shuffle=True,
            seed=42
        )

        val_generator = val_test_datagen.flow_from_directory(
            self.config.VAL_PATH,
            target_size=self.config.IMG_SIZE,
            batch_size=self.config.BATCH_SIZE,
            class_mode='binary',
            shuffle=False
        )

        test_generator = val_test_datagen.flow_from_directory(
            self.config.TEST_PATH,
            target_size=self.config.IMG_SIZE,
            batch_size=self.config.BATCH_SIZE,
            class_mode='binary',
            shuffle=False
        )

        # Generadores multi-modales CORREGIDOS
        train_multimodal = FixedMultiModalGenerator(
            train_generator, self.clinical_generator, self.config
        )
        val_multimodal = FixedMultiModalGenerator(
            val_generator, self.clinical_generator, self.config
        )
        test_multimodal = FixedMultiModalGenerator(
            test_generator, self.clinical_generator, self.config
        )

        return train_multimodal, val_multimodal, test_multimodal

def create_callbacks(config: EnhancedConfig) -> List:
    """Callbacks optimizados para CPU"""
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=config.PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=config.MIN_LR,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=f"{config.MODELS_DIR}/best_model_cpu.h5",
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    ]
    return callbacks

def train_complete_system():
    """Función principal optimizada para CPU con generador corregido"""
    
    print(" PneumoScan AI - Iniciando entrenamiento optimizado para CPU")
    print("=" * 65)
    
    # Configuración
    config = EnhancedConfig()
    config.create_directories()

    # Verificar datos
    if not os.path.exists(config.DATA_PATH):
        print(" Error: Dataset no encontrado")
        print(f"   Esperado en: {config.DATA_PATH}")
        return None

    try:
        # Pipeline de datos
        print(" Configurando pipeline de datos...")
        pipeline = DataPipeline(config)
        train_gen, val_gen, test_gen = pipeline.setup_data_generators()

        print(f"   Datos cargados:")
        print(f"   - Entrenamiento: {len(train_gen)} batches")
        print(f"   - Validación: {len(val_gen)} batches")
        print(f"   - Test: {len(test_gen)} batches")

        # Modelo
        print(" Construyendo modelo...")
        model_builder = PneumoScanMultiTaskModel(config)
        model = model_builder.create_complete_model()
        model = model_builder.compile_model(model)

        # Convertir generadores a datasets
        print(" Preparando datasets...")
        train_dataset = train_gen.get_dataset()
        val_dataset = val_gen.get_dataset()

        # Callbacks
        callbacks = create_callbacks(config)

        # Entrenamiento
        print(f" Iniciando entrenamiento ({config.EPOCHS} epochs)...")
        print(" Esto puede tomar tiempo en CPU. Ten paciencia...")
        
        start_time = datetime.now()
        
        history = model.fit(
            train_dataset,
            epochs=config.EPOCHS,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        end_time = datetime.now()
        duration = end_time - start_time

        print(f"\n Entrenamiento completado en {duration}")

        # Guardar modelo final
        model.save(f"{config.MODELS_DIR}/pneumoscan_cpu_final.h5")
        
        # Evaluación rápida
        print("\n Evaluando modelo...")
        test_dataset = test_gen.get_dataset()
        test_loss = model.evaluate(test_dataset, verbose=0)
        print(f"   - Pérdida de test: {test_loss[0]:.4f}")

        print(f"\n Modelo guardado en: {config.MODELS_DIR}/")
        print(f" Resultados en: {config.OUTPUT_BASE}/")
        
        return model, history

    except Exception as e:
        print(f" Error durante entrenamiento: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def quick_test():
    """Test rápido del sistema"""
    print(" Ejecutando test rápido del sistema...")
    
    config = EnhancedConfig()
    config.EPOCHS = 1  # Solo 1 epoch para test
    config.BATCH_SIZE = 2  # Batch muy pequeño
    
    result = train_complete_system()
    
    if result:
        print(" ¡Test exitoso! El sistema funciona correctamente.")
    else:
        print(" Test falló. Revisa los errores anteriores.")

if __name__ == "__main__":
    print(" PneumoScan AI - Sistema Optimizado para CPU Local")
    print("=" * 55)
    
    print("\n¿Qué quieres hacer?")
    print("1. Entrenamiento completo (10 epochs)")
    print("2. Test rápido (1 epoch)")
    print("3. Solo verificar configuración")
    
    try:
        choice = input("\nElige una opción (1/2/3): ").strip()
        
        if choice == "1":
            result = train_complete_system()
            if result:
                print("\n ¡Entrenamiento completo exitoso!")
            else:
                print("\n Entrenamiento falló")
                
        elif choice == "2":
            quick_test()
            
        elif choice == "3":
            config = EnhancedConfig()
            config.create_directories()
            print(" Configuración verificada")
            
        else:
            print(" Opción no válida")
            
    except KeyboardInterrupt:
        print("\n  Operación cancelada por el usuario")
    except Exception as e:
        print(f"\n Error: {str(e)}")