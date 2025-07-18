# config_local.py - Configuración optimizada para tu PC local

import os

# Optimizaciones para CPU
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '4'  # Ajusta según tus cores

# Configuración reducida para ejecución local
LOCAL_CONFIG = {
    'IMG_SIZE': (128, 128),    # Más pequeño para CPU
    'BATCH_SIZE': 4,           # Batch pequeño
    'EPOCHS': 5,               # Pocas epochs para prueba
    'LEARNING_RATE': 0.001,    # LR más alto
    'PATIENCE': 3              # Early stopping temprano
}