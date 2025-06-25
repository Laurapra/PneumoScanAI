# Pipeline Completo - Proyecto Detección de Neumonía
# Integración de todo el proyecto desde setup hasta evaluación final

from datetime import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from src.pneumonia_setup import Config, main as setup_main
from src.models import train_all_models, PneumoniaModels
from src.evaluacion_analisis import complete_evaluation_with_gradcam


print(" PROYECTO CAPSTONE: DETECCIÓN DE NEUMONÍA CON CNN")
print("=" * 60)
print("Desarrollado para análisis inteligente de radiografías de tórax")
print(f"Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)


class PneumoniaProject:
    """Clase principal que maneja todo el pipeline del proyecto"""

    def __init__(self):
        self.config = self._setup_config()
        self.train_gen = None
        self.val_gen = None
        self.test_gen = None
        self.class_weights = None
        self.model_manager = None
        self.evaluator = None
        self.results_summary = None

    def _setup_config(self):
        """Configuración del proyecto"""
        class Config:
            # Rutas de datos
            DATA_PATH = "data/chest_xray"
            TRAIN_PATH=f"{DATA_PATH}/train"
            VAL_PATH=f"{DATA_PATH}/val"
            TEST_PATH=f"{DATA_PATH}/test"
            
            #Parámetros de imagen
            IMG_SIZE = (224, 224)
            BATCH_SIZE = 32
            CHANNELS = 3
            
            #Parámetros de entrenamiento
            EPOCHS = 50
            LEARNING_RATE = 0.001
            PATIENCE = 10
            
            #Clases
            CLASSES = ['NORMAL', 'PNEUMONIA']
            NUM_CLASSES = len(CLASSES)
            
            #Configuración de modelos a entrenar
            MODELS_CONFIG = {
                'baseline': {'type': 'baseline', 'epochs': 30},
                'resnet50_frozen': {'type': 'resnet50', 'trainable_layers': 0, 'epochs': 20},
                'resnet50_ft': {'type': 'resnet50', 'trainable_layers': 20, 'epochs': 15},
                'densenet121': {'type': 'densenet121', 'trainable_layers': 0, 'epochs': 20},
                'inceptionv3': {'type': 'inceptionv3', 'trainable_layers': 0, 'epochs': 20}
            }
        
        return Config()
    
    def run_complete_pipeline(self, skip_training=False):
        """Ejecutar el pipeline completo del proyecto"""
        
        print("\n INICIANDO PIPELINE COMPLETO")
        print("=" * 50)
        
        #Paso 1: Setup inicial
        print("\n PASO 1: CONFIGURACIÓN INICIAL")
        success = self._setup_project()
        if not success:
            print(" Error en la configuración inicial. Abortando.")
            return False
        
        #Paso 2: Análisis exploratorio
        print("\n PASO 2: ANÁLISIS EXPLORATORIO DE DATOS")
        self._perform_eda()
        
        #Paso 3: Preparación de datos
        print("\n PASO 3: PREPARACIÓN DE DATOS")
        self._prepare_data()
        
        #Paso 4: Entrenamiento de modelos
        if not skip_training:
            print("\n PASO 4: ENTRENAMIENTO DE MODELOS")
            self._train_models()
        else:
            print("\n⏭  PASO 4: SALTANDO ENTRENAMIENTO (usando modelos existentes)")
            self._load_existing_models()
        
        # Paso 5: Evaluación y análisis
        print("\n PASO 5: EVALUACIÓN Y ANÁLISIS")
        self._evaluate_models()
        
        # Paso 6: Informe final
        print("\n PASO 6: GENERACIÓN DE INFORME FINAL")
        self._generate_final_report()
        
        print("\n PIPELINE COMPLETADO EXITOSAMENTE")
        return True
    
    def _setup_project(self):
        """Configuración inicial del proyecto"""
        
        #Crear directorios
        directories = ['models', 'results', 'plots', 'logs', 'reports']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Directorio '{directory}' creado/verificado")
        
        #Verificar dataset
        if not os.path.exists(self.config.DATA_PATH):
            print(f" Dataset no encontrado en: {self.config.DATA_PATH}")
            print("\n INSTRUCCIONES PARA DESCARGAR EL DATASET:")
            print("1. Ir a: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
            print("2. Descargar el archivo ZIP")
            print("3. Extraer en la carpeta del proyecto como 'chest_xray/'")
            print("4. Verificar que existan las carpetas: train/, val/, test/")
            return False
        
        #Verificar estructura
        required_dirs = ['train', 'val', 'test']
        for subdir in required_dirs:
            path = os.path.join(self.config.DATA_PATH, subdir)
            if not os.path.exists(path):
                print(f" Directorio faltante: {path}")
                return False
            
            for class_name in self.config.CLASSES:
                class_path = os.path.join(path, class_name)
                if not os.path.exists(class_path):
                    print(f" Clase faltante: {class_path}")
                    return False
        
        print(" Estructura del dataset verificada correctamente")
        return True
    
    def _perform_eda(self):
        """Análisis exploratorio de datos"""
        
        #Contar imágenes
        data_info = {}
        for split in ['train', 'val', 'test']:
            data_info[split] = {}
            for class_name in self.config.CLASSES:
                class_path = os.path.join(self.config.DATA_PATH, split, class_name)
                count = len([f for f in os.listdir(class_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                data_info[split][class_name] = count
        
        #Crear DataFrame
        df_counts = pd.DataFrame(data_info).T
        print("\n Distribución de imágenes por conjunto:")
        print(df_counts)
        
        #Estadísticas
        total_normal = sum(df_counts['NORMAL'])
        total_pneumonia = sum(df_counts['PNEUMONIA'])
        total = total_normal + total_pneumonia
        
        print(f"\n Estadísticas generales:")
        print(f"Total: {total:,} imágenes")
        print(f"Normal: {total_normal:,} ({total_normal/total*100:.1f}%)")
        print(f"Neumonía: {total_pneumonia:,} ({total_pneumonia/total*100:.1f}%)")
        print(f"Ratio Neumonía/Normal: {total_pneumonia/total_normal:.2f}")
        
        #Guardar estadísticas
        df_counts.to_csv('results/dataset_statistics.csv')
        
        # Detectar desequilibrio
        if total_pneumonia > total_normal * 1.5:
            print("\n  DESEQUILIBRIO DE CLASES DETECTADO")
            print("   Estrategias implementadas:")
            print("    Pesos de clase balanceados")
            print("    Data augmentation")
            print("    Métricas especializadas (F1, AUC)")
        
        return df_counts
    
    def _prepare_data(self):
        """Preparar generadores de datos"""
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from sklearn.utils.class_weight import compute_class_weight
        
        #Generador de entrenamiento con augmentación
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        #Generadores de validación y test (solo normalización)
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        #Crear generadores
        self.train_gen = train_datagen.flow_from_directory(
            self.config.TRAIN_PATH,
            target_size=self.config.IMG_SIZE,
            batch_size=self.config.BATCH_SIZE,
            class_mode='binary',
            shuffle=True,
            seed=42
        )
        
        self.val_gen = val_test_datagen.flow_from_directory(
            self.config.VAL_PATH,
            target_size=self.config.IMG_SIZE,
            batch_size=self.config.BATCH_SIZE,
            class_mode='binary',
            shuffle=False
        )
        
        self.test_gen = val_test_datagen.flow_from_directory(
            self.config.TEST_PATH,
            target_size=self.config.IMG_SIZE,
            batch_size=self.config.BATCH_SIZE,
            class_mode='binary',
            shuffle=False
        )
        
        #Calcular pesos de clase
        y_train = self.train_gen.classes
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        self.class_weights = dict(zip(np.unique(y_train), class_weights))
        
        print(f"  Generadores creados:")
        print(f"  Entrenamiento: {self.train_gen.samples:,} imágenes")
        print(f"  Validación: {self.val_gen.samples:,} imágenes")
        print(f"  Prueba: {self.test_gen.samples:,} imágenes")
        print(f"✓ Pesos de clase: {self.class_weights}")
    
    def _train_models(self):
        """Entrenar todos los modelos configurados"""
        from tensorflow.keras import layers, Sequential
        from tensorflow.keras.applications import ResNet50, DenseNet121, InceptionV3
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        
        print(f" Entrenando {len(self.config.MODELS_CONFIG)} modelos...")
        
        self.model_manager = PneumoniaModels(self.config)
        training_results = {}
        
        for model_name, model_config in self.config.MODELS_CONFIG.items():
            print(f"\n{'='*60}")
            print(f" ENTRENANDO: {model_name.upper()}")
            print(f"{'='*60}")
            
            #Crear modelo según configuración
            if model_config['type'] == 'baseline':
                model = self._create_baseline_model()
            elif model_config['type'] == 'resnet50':
                model = self._create_resnet_model(model_config.get('trainable_layers', 0))
            elif model_config['type'] == 'densenet121':
                model = self._create_densenet_model(model_config.get('trainable_layers', 0))
            elif model_config['type'] == 'inceptionv3':
                model = self._create_inception_model(model_config.get('trainable_layers', 0))
            
            #Guardar modelo
            self.model_manager.models[model_name] = model
            
            #Configurar callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7),
                ModelCheckpoint(f'models/{model_name}_best.h5', monitor='val_loss', save_best_only=True)
            ]
            
            #Entrenar
            history = model.fit(
                self.train_gen,
                epochs=model_config['epochs'],
                validation_data=self.val_gen,
                class_weight=self.class_weights,
                callbacks=callbacks,
                verbose=1
            )
            
            #Guardar historial
            self.model_manager.histories[model_name] = history
            training_results[model_name] = {
                'final_val_loss': history.history['val_loss'][-1],
                'final_val_acc': history.history['val_accuracy'][-1],
                'epochs_trained': len(history.history['loss'])
            }
            
            print(f"   {model_name} completado")
            print(f"   Val Loss: {training_results[model_name]['final_val_loss']:.4f}")
            print(f"   Val Acc: {training_results[model_name]['final_val_acc']:.4f}")
        
        # Guardar resumen de entrenamiento
        pd.DataFrame(training_results).T.to_csv('results/training_summary.csv')
        print(f"\n Entrenamiento completado para todos los modelos")
    
    def _create_baseline_model(self):
        """Crear modelo CNN baseline"""
        model = Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.config.IMG_SIZE, 3)),
            layers.MaxPooling2D(2, 2),
            layers.BatchNormalization(),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.BatchNormalization(),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.BatchNormalization(),
            
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.BatchNormalization(),
            
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.config.LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        return model
    
    def _create_resnet_model(self, trainable_layers=0):
        """Crear modelo ResNet50"""
        base_model = ResNet50(weights='imagenet', include_top=False, 
                             input_shape=(*self.config.IMG_SIZE, 3))
        
        base_model.trainable = trainable_layers > 0
        if trainable_layers > 0:
            for layer in base_model.layers[:-trainable_layers]:
                layer.trainable = False
        
        model = Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        lr = self.config.LEARNING_RATE * 0.1 if trainable_layers == 0 else self.config.LEARNING_RATE * 0.01
        model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy',
                     metrics=['accuracy', 'precision', 'recall'])
        return model
    
    def _create_densenet_model(self, trainable_layers=0):
        """Crear modelo DenseNet121"""
        base_model = DenseNet121(weights='imagenet', include_top=False,
                                input_shape=(*self.config.IMG_SIZE, 3))
        
        base_model.trainable = trainable_layers > 0
        if trainable_layers > 0:
            for layer in base_model.layers[:-trainable_layers]:
                layer.trainable = False
        
        model = Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        lr = self.config.LEARNING_RATE * 0.1 if trainable_layers == 0 else self.config.LEARNING_RATE * 0.01
        model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy',
                     metrics=['accuracy', 'precision', 'recall'])
        return model
    
    def _create_inception_model(self, trainable_layers=0):
        """Crear modelo InceptionV3"""
        base_model = InceptionV3(weights='imagenet', include_top=False,
                                input_shape=(*self.config.IMG_SIZE, 3))
        
        base_model.trainable = trainable_layers > 0
        if trainable_layers > 0:
            for layer in base_model.layers[:-trainable_layers]:
                layer.trainable = False
        
        model = Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        lr = self.config.LEARNING_RATE * 0.1 if trainable_layers == 0 else self.config.LEARNING_RATE * 0.01
        model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy',
                     metrics=['accuracy', 'precision', 'recall'])
        return model
    
    def _load_existing_models(self):
        """Cargar modelos previamente entrenados"""
        print(" Cargando modelos existentes...")
        
        self.model_manager = PneumoniaModels(self.config)
        
        for model_name in self.config.MODELS_CONFIG.keys():
            model_path = f'models/{model_name}_best.h5'
            if os.path.exists(model_path):
                try:
                    model = tf.keras.models.load_model(model_path)
                    self.model_manager.models[model_name] = model
                    print(f" {model_name} cargado desde {model_path}")
                except Exception as e:
                    print(f" Error cargando {model_name}: {e}")
            else:
                print(f"  Modelo {model_name} no encontrado en {model_path}")
    
    def _evaluate_models(self):
        """Evaluar todos los modelos"""
        from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
        
        print(" Iniciando evaluación de modelos...")
        
        self.evaluator = PneumoniaEvaluator(self.config)
        evaluation_results = {}
        
        for model_name, model in self.model_manager.models.items():
            print(f"\n Evaluando: {model_name}")
            
            #Resetear generador
            self.test_gen.reset()
            
            #Predicciones
            predictions = model.predict(self.test_gen, verbose=0)
            y_pred = (predictions > 0.5).astype(int).flatten()
            y_pred_proba = predictions.flatten()
            y_true = self.test_gen.classes
            
            #Métricas
            from sklearn.metrics import accuracy_score, f1_score
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            
            #ROC AUC
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            #Matriz de confusión
            cm = confusion_matrix(y_true, y_pred)
            
            #Guardar resultados
            evaluation_results[model_name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'confusion_matrix': cm.tolist(),
                'predictions': y_pred.tolist(),
                'probabilities': y_pred_proba.tolist()
            }
            
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   F1-Score: {f1:.4f}")
            print(f"   ROC AUC: {roc_auc:.4f}")
        
        #Crear resumen comparativo
        results_df = pd.DataFrame({
            model: {'Accuracy': results['accuracy'], 
                   'F1-Score': results['f1_score'],
                   'ROC AUC': results['roc_auc']}
            for model, results in evaluation_results.items()
        }).T
        
        self.results_summary = results_df
        results_df.to_csv('results/evaluation_summary.csv')
        
        print(f"\n RESUMEN DE EVALUACIÓN:")
        print(results_df.round(4))
        
        # Identificar mejor modelo
        best_model = results_df['F1-Score'].idxmax()
        print(f"\n MEJOR MODELO: {best_model}")
        print(f"   F1-Score: {results_df.loc[best_model, 'F1-Score']:.4f}")
        print(f"   Accuracy: {results_df.loc[best_model, 'Accuracy']:.4f}")
        print(f"   ROC AUC: {results_df.loc[best_model, 'ROC AUC']:.4f}")
        
        return evaluation_results
    
    def _generate_final_report(self):
        """Generar informe final del proyecto"""
        
        report_content = f"""
# INFORME TÉCNICO: DETECCIÓN DE NEUMONÍA CON CNN

**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Proyecto:** Capstone - Detección Inteligente de Neumonía

## 1. RESUMEN EJECUTIVO

Este proyecto desarrolló un sistema de inteligencia artificial para la detección automática de neumonía en radiografías de tórax utilizando Redes Neuronales Convolucionales (CNN). Se implementaron y compararon múltiples arquitecturas, incluyendo un modelo baseline y modelos basados en transfer learning.

## 2. DATASET

- **Fuente:** Chest X-Ray Images (Pneumonia) - Kaggle
- **Total de imágenes:** {self.train_gen.samples + self.val_gen.samples + self.test_gen.samples:,}
- **Distribución:**
  - Entrenamiento: {self.train_gen.samples:,} imágenes
  - Validación: {self.val_gen.samples:,} imágenes  
  - Prueba: {self.test_gen.samples:,} imágenes

## 3. MODELOS IMPLEMENTADOS

{chr(10).join([f"- **{name}:** {config['type']} ({'frozen' if config.get('trainable_layers', 0) == 0 else 'fine-tuned'})" for name, config in self.config.MODELS_CONFIG.items()])}

## 4. RESULTADOS PRINCIPALES

### Comparación de Modelos:
```
{self.results_summary.round(4).to_string()}
```

### Mejor Modelo: {self.results_summary['F1-Score'].idxmax()}
- **F1-Score:** {self.results_summary['F1-Score'].max():.4f}
- **Accuracy:** {self.results_summary.loc[self.results_summary['F1-Score'].idxmax(), 'Accuracy']:.4f}
- **ROC AUC:** {self.results_summary.loc[self.results_summary['F1-Score'].idxmax(), 'ROC AUC']:.4f}

## 5. IMPLICACIONES CLÍNICAS

El modelo desarrollado demuestra capacidad para asistir en el diagnóstico de neumonía, con potencial para:
- Acelerar el proceso diagnóstico
- Reducir la variabilidad entre observadores
- Proporcionar una segunda opinión automática
- Mejorar la eficiencia en entornos clínicos

## 6. LIMITACIONES Y RECOMENDACIONES FUTURAS

### Limitaciones:
- Dependencia de la calidad de las imágenes de entrenamiento
- Necesidad de validación clínica adicional
- Posibles sesgos en el dataset

### Recomendaciones:
- Validación con datasets externos
- Implementación de técnicas de explicabilidad (Grad-CAM)
- Evaluación con radiólogos expertos
- Desarrollo de interfaz clínica

## 7. CONCLUSIONES

El proyecto demuestra la viabilidad de utilizar deep learning para la detección de neumonía en radiografías, con resultados prometedores que justifican investigación adicional y posible implementación clínica con las validaciones apropiadas.

---
*Informe generado automáticamente por el sistema de evaluación*
"""
        
        #Guardar informe
        with open('reports/informe_final.md', 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(" Informe final generado: 'reports/informe_final.md'")
        
        #Crear visualizaciones finales
        self._create_final_visualizations()
    
    def _create_final_visualizations(self):
        """Crear visualizaciones finales para el informe"""
        
        #Gráfico comparativo de modelos
        plt.figure(figsize=(12, 8))
        
        #Subplot 1: Métricas comparativas
        plt.subplot(2, 2, 1)
        metrics = ['Accuracy', 'F1-Score', 'ROC AUC']
        x = np.arange(len(self.results_summary.index))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            plt.bar(x + i*width, self.results_summary[metric], width, label=metric)
        
        plt.xlabel('Modelos')
        plt.ylabel('Score')
        plt.title('Comparación de Métricas por Modelo')
        plt.xticks(x + width, self.results_summary.index, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        #Subplot 2: Ranking de modelos
        plt.subplot(2, 2, 2)
        best_models = self.results_summary.sort_values('F1-Score', ascending=True)
        plt.barh(range(len(best_models)), best_models['F1-Score'])
        plt.yticks(range(len(best_models)), best_models.index)
        plt.xlabel('F1-Score')
        plt.title('Ranking de Modelos por F1-Score')
        plt.grid(True, alpha=0.3)
        
        #Subplot 3: Distribución de accuracy
        plt.subplot(2, 2, 3)
        plt.hist(self.results_summary['Accuracy'], bins=10, alpha=0.7, edgecolor='black')
        plt.xlabel('Accuracy')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de Accuracy')
        plt.grid(True, alpha=0.3)
        
        #subplot 4: Correlación entre métricas
        plt.subplot(2, 2, 4)
        plt.scatter(self.results_summary['Accuracy'], self.results_summary['F1-Score'])
        plt.xlabel('Accuracy')
        plt.ylabel('F1-Score')
        plt.title('Correlación Accuracy vs F1-Score')
        
        #Añadir etiquetas a los puntos
        for i, model in enumerate(self.results_summary.index):
            plt.annotate(model, 
                        (self.results_summary['Accuracy'][i], self.results_summary['F1-Score'][i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/final_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(" Visualizaciones finales guardadas en 'plots/'")

#Clase auxiliar para modelos (simplificada para el pipeline completo)
class PneumoniaModels:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.histories = {}

#Clase auxiliar para evaluación (simplificada para el pipeline completo)  
class PneumoniaEvaluator:
    def __init__(self, config):
        self.config = config
        self.results = {}

#Función principal para ejecutar todo
def main():
    """Función principal para ejecutar el proyecto completo"""
    
    #crear instancia del proyecto
    project = PneumoniaProject()
    
    #ejecutar pipeline completo
    #Cambiar skip_training=True si quieres cargar modelos existentes
    success = project.run_complete_pipeline(skip_training=False)
    
    if success:
        print("\n ¡PROYECTO COMPLETADO EXITOSAMENTE!")
        print("\n Archivos generados:")
        print("    results/dataset_statistics.csv")
        print("    results/training_summary.csv") 
        print("    results/evaluation_summary.csv")
        print("    models/*.h5 (modelos entrenados)")
        print("    plots/*.png (visualizaciones)")
        print("    reports/informe_final.md")
        
        print(f"\n Mejor modelo: {project.results_summary['F1-Score'].idxmax()}")
        print(f"   F1-Score: {project.results_summary['F1-Score'].max():.4f}")
        
    else:
        print("\n Error durante la ejecución del proyecto")
    
    return project

if __name__ == "__main__":
    #Configurar TensorFlow
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU disponible: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    #ejecutar proyecto
    project = main()