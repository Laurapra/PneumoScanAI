#Proyecto Capstone: Detección de Neumonía
#Configuración inicial y estructura del proyecto

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

#Biblioteca para procesamiento de imágenes
import cv2
import PIL as Image
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight

# TensorFlow y Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, DenseNet121, InceptionV3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

#configuración de reproducibilidad
SEED=42
np.random.seed(SEED)
tf.random.set_seed(SEED)

#configuración del proyecto
class Config:
  #rutas de trabajo
  DATA_PATH="data/chest_xray"
  TRAIN_PATH=f"{DATA_PATH}/train"
  VAL_PATH=f"{DATA_PATH}/val"
  TEST_PATH=f"{DATA_PATH}/test"
  
  #parámetros de imagen
  IMG_SIZE=(224,224) #tamaño estándar para modelos pre-entrenados
  BATCH_SIZE=32
  CHANNELS=3
  
  #parámetros de entrenamiento
  EPOCHS=50
  LEARNING_RATE=0.001
  PATIENCE=10
  
  #clases
  CLASSES=['NORMAL', 'PNEUMONIA']
  NUM_CLASSES=len(CLASSES)
config=Config()

def setup_directories():
  """Crear estructura de directorios para el proyecto"""
  directories=[
    'models',
    'results',
    'plots',
    'logs',
    'notebooks'
  ]
  for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"Directorio creado: '{directory}' creado/verificado")

def check_dataset_structure(data_path):
  """Verificar la estructura del dataset"""
  print("VERIFICACIÓN DE ESTRUCTURA DEL DATASET")
  
  if not os.path.exists(data_path):
    print(f"Error: No se encuentra el directorio {data_path}")
    print("INSTRUCCIONES para descargar el dataset:")
    print("1. Ir a: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data")
    print("2. Descargar el archivo ZIP")
    print("3. Extraer en la carpeta del proyecto")
    return False

  #verifico los subdirectorios
  subdirs=['train','val','test']
  for subdir in subdirs:
    subdir_path=os.path.join(data_path, subdir)
    if os.path.exists(subdir_path):
      print(f"{subdir}/encontrado")
      
      #verifico clases en cada subdirectorio
      for class_name in config.CLASSES:
        class_path=os.path.join(subdir_path, class_name)
        if os.path.exists(class_path):
          count=len(os.listdir(class_path))
          print(f"-{class_name}: {count} imágenes")
        else:
          print(f"{class_name}: No encontrado")
    else:
      print(f"{subdir}/No encontrado")
  return True

def perform_eda():
    """Análisis exploratorio de datos"""
    print("\n=== ANÁLISIS EXPLORATORIO DE DATOS ===")
    
    # Contar imágenes por conjunto y clase
    data_info = {}
    
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(config.DATA_PATH, split)
        data_info[split] = {}
        
        for class_name in config.CLASSES:
            class_path = os.path.join(split_path, class_name)
            if os.path.exists(class_path):
                count = len([f for f in os.listdir(class_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                data_info[split][class_name] = count
            else:
                data_info[split][class_name] = 0
    
    # Crear DataFrame para visualización
    df_counts = pd.DataFrame(data_info).T
    print("\n📊 Distribución de imágenes:")
    print(df_counts)
    
    # Calcular estadísticas
    total_normal = sum(df_counts['NORMAL'])
    total_pneumonia = sum(df_counts['PNEUMONIA'])
    total_images = total_normal + total_pneumonia
    
    print(f"\n📈 Estadísticas generales:")
    print(f"Total de imágenes: {total_images}")
    print(f"Normal: {total_normal} ({total_normal/total_images*100:.1f}%)")
    print(f"Neumonía: {total_pneumonia} ({total_pneumonia/total_images*100:.1f}%)")
    print(f"Ratio Neumonía/Normal: {total_pneumonia/total_normal:.2f}")
    
    # Identificar desequilibrio de clases
    if total_pneumonia > total_normal * 1.5:
        print("\n  DESEQUILIBRIO DE CLASES DETECTADO")
        print("   Se necesitarán técnicas de balanceamiento:")
        print("   - Class weighting")
        print("   - Data augmentation")
        print("   - Focal loss")
    
    return df_counts

def visualize_sample_images():
  """Visualizar imágenes de muestra"""
  print("VISUALIZACIÓN DE IMÁGENES DE MUESTRA")
  
  fig, axes=plt.subplots(2,4,figsize=(15,8))
  fig.suptitle('Ejemplos de Radiografías de Tórax', fontsize=16)
  
  for i, class_name in enumerate(config.CLASSES):
    class_path=os.path.join(config.TRAIN_PATH, class_name)
    if os.path.exists(class_path):
      #tomar las 4 primeras imágenes de cada clase
      images=[f for f in os.listdir(class_path)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:4]
      for j, img_name in enumerate(images):
        img_path=os.path.join(class_path, img_name)
        img=cv2.imread(img_path)
        if img is not None:
          img_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
          axes[i,j].imshow(img_rgb, cmap='gray')
          axes[i,j].set_title(f'{class_name}')
          axes[i,j].axis('off')
      plt.tight_layout()
      plt.savefig('plots/sample_images.png', dpi=300, bbox_inches='tight')
      plt.show()
      
def create_data_generators():
  """Crear generadores de datos con argumentación"""
  print("CREACIÓN DE GENERADORES DE DATOS")
  
  #generador para entrenamiento (con augmentación)
  train_datage= ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
  )
  
  #generador para validación y test (solo normalización)
  val_test_datagen= ImageDataGenerator(rescale=1./255)
  #crear generadores
  train_generator=train_datage.flow_from_directory(
    config.TRAIN_PATH,
    target_size=config.IMG_SIZE,
    batch_size=config.BATCH_SIZE,
    class_mode='binary',
    shuffle=True,
    seed=SEED
  )
  val_generator=val_test_datagen.flow_from_directory(
    config.VAL_PATH,
    target_size=config.IMG_SIZE,
    batch_size=config.BATCH_SIZE,
    class_mode='binary',
    shuffle=False
  )
  test_generator=val_test_datagen.flow_from_directory(
    config.TEST_PATH,
    target_size=config.IMG_SIZE,
    batch_size=config.BATCH_SIZE,
    class_mode='binary',
    shuffle=False
  )
  print(f"Generador de entrenamiento: {train_generator.samples} imágenes")
  print(f"Generador de validación: {val_generator.samples} imágenes")
  print(f"Generador de prueba: {test_generator.samples} imágenes")
  print(f"Mapeo de clases: {train_generator.class_indices}")
  
  return train_generator, val_generator, test_generator

def calculate_class_weights(train_generator):
  """Calcular pesos de clase para manejar el desequilibrio"""
  print("CÁLCULO DE PESOS DE CLASE")
  
  #obtener las etiquetas del generador de entrenamiento
  y_train=train_generator.classes
  #calcular pesos de clase
  class_weights=compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
  )
  class_weight_dict=dict(zip(np.unique(y_train), class_weights))
  print(f"Pesos de clase calculados: {class_weight_dict}")
  print("Esto ayudará a balancear el entrenamiento hacia la clase minoritatia")
  return class_weight_dict

#función principal para ejecutar la configuración inicial
def main():
  """Función principal para ejecutar la configuración inicial"""
  print("🚀 INICIANDO PneumoScanAI")
  print("="*50)
  
  #1.configurar directorios
  setup_directories()
  #2.verificar dataset
  if not check_dataset_structure(config.DATA_PATH):
    return
  #3.análisis exploratorio
  df_counts=perform_eda()
  #4.visualizar muestras
  visualize_sample_images()
  #5.crear generadores de datos
  train_gen, val_gen, test_gen=create_data_generators()
  #6.calcular pesos de clase
  class_weights=calculate_class_weights(train_gen)
  
  print("CONFIGURACIÓN INICIAL COMPLETADA")
  print("Próximos pasos:")
  print("1. Implementar modelo baseline (CNN Simple)")
  print("2. Implementar transfer learning (ResNet, DenseNet)")
  print("3. Entrenat y evaluar modelos")
  print("4. Análisis de resultados y visualización")
  
  return train_gen, val_gen, test_gen, class_weights

if __name__=='__main__':
  #ejecutar configuración inicial
  generators=main()