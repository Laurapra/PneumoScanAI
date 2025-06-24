# 🧠 Proyecto Capstone: Detección Inteligente de Neumonía (PneumoScanAI)

## 📋 Descripción del Proyecto
Este proyecto implementa un **sistema de inteligencia artificial completo** para la detección automática de neumonía en radiografías de tórax utilizando **Redes Neuronales Convolucionales (CNN)** con análisis de explicabilidad mediante **Grad-CAM**. El sistema está diseñado para asistir a profesionales de la salud en el diagnóstico y preciso de neumonía.

## 🎯Objetivos Principales
- **Detección Automática**: Clasificar radiografías como normales o con neumonía.
- **Alta Precisión**: Lograr métricas clínicamente relevantes (>90% sensibilidad).
- **Explicabilidad**: Mostrar qué áreas de la imagen influyen en la decisión.
- **Aplicación Práctica**: Preparado para implementación en entornos clínicos. 
  
## 🚀Características Principales
✅ Modelos Implementados

- CNN Baseline: Arquitectura personalizada de 4 capas convolucionales
- ResNet50: Transfer learning con ImageNet (frozen & fine-tuned)
- DenseNet121: Arquitectura densa eficiente
- InceptionV3: Módulos inception para captura multi-escala
- Comparación Automática: Evaluación y selección del mejor modelo

✅ Análisis de Explicabilidad

- Grad-CAM Integrado: Visualización de atención del modelo
- Análisis de Errores: Explicabilidad específica para falsos positivos/negativos
- Interpretación Clínica: Guías para entender las decisiones del modelo
- Mapas de Calor: Visualización intuitiva de regiones importantes

✅ Pipeline Completo

- EDA Avanzado: Análisis exploratorio con visualizaciones profesionales
- Manejo de Desequilibrio: Class weighting + data augmentation
- Evaluación Clínica: Métricas relevantes para uso médico
- Automatización: Ejecución completa con un solo comando

✅ Preparado para Producción

- Interfaz Gráfica: Streamlit app lista para deployment
- Código Modular: Estructura profesional y reutilizable
- Documentación Completa: Guías de uso e interpretación
- Reproducibilidad: Seeds fijos y configuración estable

📊 Resultados Destacados


🏆 Mejor modelo: 

## 🛠️ Tecnologías Utilizadas
**Core Technologies**

- TensorFlow/Keras - Deep learning framework
- OpenCV - Procesamiento de imágenes
- Scikit-learn - Métricas y evaluación
- Pandas/NumPy - Manipulación de datos
- Matplotlib/Seaborn - Visualizaciones

**Explicabilidad**

- Grad-CAM - Gradient-weighted Class Activation Mapping
- Custom Visualization - Mapas de calor e interpretación

## 🚀 **Instalación y Configuración**
**Requisitos del Sistema**
- Python 3.8 o Superior
- GPU recomendada (CUDA compatible) - opcional
- 8GB RAM mínimo, 16GB recomendado
- 10GB espacio libre en disco
  1. **Clonar el Repositorio**
     git clone https://github.com/tu-usuario/capstone-neumonia.git
      cd PNEUMOSCANIA
  2. **Crear Entorno Virtual**
     # Crear entorno virtual
     python -m venv pneumonia_env

    # Activar entorno (Windows)
    pneumonia_env\Scripts\activate

    # Activar entorno (Mac/Linux)
    source pneumonia_env/bin/activate
  3. **Instalar Dependencias**
    # Instalar dependencias principales
    pip install -r requirements.txt

    # Verificar instalación de TensorFlow
    python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPU:', len(tf.config.list_physical_devices('GPU')) > 0)"
  4. **Descargar el Dataset**
    a. **Ir a Kaggle**: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data
## 🎯 **Uso del Sistema**
🚀 **Ejecución Completa (Recomendado)
# Ejecutar pipeline completo automático
python src/pneumonia_complete_pipeline.py
**Esto ejecutará automáticamente**:
1. Configuración inicial y verificación del dataset
2. Análisis exploratorio de datos (EDA) avanzado
3. Preparación de generadores con data augmentation
4. Entrenamiento de todos los modelos (baseline + transfer learning)
5. Evaluación completa con métricas clínicas
6. Análisis Grad-CAM para explicabilidad
7. Generación de informe final automático
