#Evaluación y Análisis de Resultados - Detección de Neumonía
#Métricas, visualizaciones y análisis de errores + GRAD-CAM INTEGRADO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc, 
    precision_recall_curve, f1_score, accuracy_score
)
import tensorflow as tf
from tensorflow import keras
import cv2
from pathlib import Path
import os

class PneumoniaEvaluator:
    """Clase para evaluar modelos de detección de neumonía CON GRAD-CAM"""
    
    def __init__(self, config):
        self.config = config
        self.results = {}
        
    def evaluate_model(self, model, test_generator, model_name):
        """Evaluar un modelo en el conjunto de prueba"""
        print(f"\n EVALUANDO MODELO: {model_name}")
        print("=" * 50)
        
        #Resetear generador
        test_generator.reset()
        
        #Obtener predicciones
        print("Generando predicciones...")
        predictions = model.predict(test_generator, verbose=1)
        y_pred = (predictions > 0.5).astype(int).flatten()
        y_pred_proba = predictions.flatten()
        
        #Obtener etiquetas verdaderas
        y_true = test_generator.classes
        
        #Calcular métricas
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        #Report detallado
        report = classification_report(y_true, y_pred, 
                                     target_names=['Normal', 'Neumonía'], 
                                     output_dict=True)
        
        #Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        
        #ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        #Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        #Guardar resultados
        self.results[model_name] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'true_labels': y_true,
            'fpr': fpr,
            'tpr': tpr,
            'precision': precision,
            'recall': recall
        }
        
        #Mostrar métricas principales
        print(f" Resultados para {model_name}:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   ROC AUC: {roc_auc:.4f}")
        print(f"   PR AUC: {pr_auc:.4f}")
        
        # Métricas por clase
        print(f"\n Métricas por clase:")
        for class_name, metrics in report.items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                print(f"   {class_name}:")
                print(f"     Precision: {metrics['precision']:.4f}")
                print(f"     Recall: {metrics['recall']:.4f}")
                print(f"     F1-Score: {metrics['f1-score']:.4f}")
        
        return self.results[model_name]

    #TUS MÉTODOS ORIGINALES (SIN CAMBIOS)
    
    def plot_confusion_matrices(self, models_to_plot=None):
        """Visualizar matrices de confusión"""
        if models_to_plot is None:
            models_to_plot = list(self.results.keys())
        
        n_models = len(models_to_plot)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if hasattr(axes, '__len__') else [axes]
        else:
            axes = axes.flatten()
        
        for i, model_name in enumerate(models_to_plot):
            if model_name not in self.results:
                continue
                
            cm = self.results[model_name]['confusion_matrix']
            
            #Normalizar matriz de confusión
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            #Crear heatmap
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=['Normal', 'Neumonía'],
                       yticklabels=['Normal', 'Neumonía'],
                       ax=axes[i])
            
            axes[i].set_title(f'{model_name}\nAccuracy: {self.results[model_name]["accuracy"]:.3f}')
            axes[i].set_xlabel('Predicción')
            axes[i].set_ylabel('Verdad')
        
        #Ocultar ejes sobrantes
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('plots/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, models_to_plot=None):
        """Visualizar curvas ROC"""
        if models_to_plot is None:
            models_to_plot = list(self.results.keys())
        
        plt.figure(figsize=(10, 8))
        
        for model_name in models_to_plot:
            if model_name not in self.results:
                continue
                
            fpr = self.results[model_name]['fpr']
            tpr = self.results[model_name]['tpr']
            roc_auc = self.results[model_name]['roc_auc']
            
            plt.plot(fpr, tpr, linewidth=2, 
                    label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        #Línea diagonal (clasificador aleatorio)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Aleatorio (AUC = 0.500)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title('Curvas ROC - Comparación de Modelos')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.savefig('plots/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curves(self, models_to_plot=None):
        """Visualizar curvas Precision-Recall"""
        if models_to_plot is None:
            models_to_plot = list(self.results.keys())
        
        plt.figure(figsize=(10, 8))
        
        for model_name in models_to_plot:
            if model_name not in self.results:
                continue
                
            precision = self.results[model_name]['precision']
            recall = self.results[model_name]['recall']
            pr_auc = self.results[model_name]['pr_auc']
            
            plt.plot(recall, precision, linewidth=2,
                    label=f'{model_name} (AUC = {pr_auc:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Curvas Precision-Recall - Comparación de Modelos')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        plt.savefig('plots/precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_errors(self, model_name, test_generator, n_examples=8):
        """Analizar errores del modelo (falsos positivos y negativos)"""
        if model_name not in self.results:
            print(f" No hay resultados para el modelo '{model_name}'")
            return
        
        print(f"\n ANÁLISIS DE ERRORES - {model_name}")
        print("=" * 50)
        
        #Obtener datos
        y_true = self.results[model_name]['true_labels']
        y_pred = self.results[model_name]['predictions']
        y_proba = self.results[model_name]['probabilities']
        
        #Identificar errores
        false_positives = np.where((y_true == 0) & (y_pred == 1))[0]
        false_negatives = np.where((y_true == 1) & (y_pred == 0))[0]
        
        print(f"   Resumen de errores:")
        print(f"   Falsos Positivos: {len(false_positives)}")
        print(f"   Falsos Negativos: {len(false_negatives)}")
        print(f"   Total de errores: {len(false_positives) + len(false_negatives)}")
        
        #Mostrar ejemplos de errores
        self._show_error_examples(test_generator, false_positives, false_negatives, 
                                y_proba, n_examples)
        
        #Análisis de confianza en errores
        self._analyze_error_confidence(false_positives, false_negatives, y_proba)
        
        return false_positives, false_negatives
    
    def _show_error_examples(self, test_generator, false_positives, false_negatives, 
                           y_proba, n_examples):
        """Mostrar ejemplos de errores"""
        
        #Seleccionar ejemplos aleatorios
        fp_examples = np.random.choice(false_positives, 
                                     min(n_examples//2, len(false_positives)), 
                                     replace=False) if len(false_positives) > 0 else []
        fn_examples = np.random.choice(false_negatives, 
                                     min(n_examples//2, len(false_negatives)), 
                                     replace=False) if len(false_negatives) > 0 else []
        
        fig, axes = plt.subplots(2, max(len(fp_examples), len(fn_examples)), 
                               figsize=(4*max(len(fp_examples), len(fn_examples)), 8))
        
        if len(fp_examples) == 0 and len(fn_examples) == 0:
            print(" ¡No hay errores para mostrar!")
            return
        
        #Obtener imágenes del generador
        test_generator.reset()
        images = []
        for i in range(len(test_generator)):
            batch_images, _ = test_generator[i]
            images.extend(batch_images)
            if len(images) >= test_generator.samples:
                break
        images = np.array(images[:test_generator.samples])
        
        #Mostrar falsos positivos
        for i, idx in enumerate(fp_examples):
            if i < axes.shape[1]:
                axes[0, i].imshow(images[idx])
                axes[0, i].set_title(f'Falso Positivo\nConfianza: {y_proba[idx]:.3f}')
                axes[0, i].axis('off')
        
        #Ocultar ejes sobrantes en la primera fila
        for i in range(len(fp_examples), axes.shape[1]):
            axes[0, i].set_visible(False)
        
        #Mostrar falsos negativos
        for i, idx in enumerate(fn_examples):
            if i < axes.shape[1]:
                axes[1, i].imshow(images[idx])
                axes[1, i].set_title(f'Falso Negativo\nConfianza: {y_proba[idx]:.3f}')
                axes[1, i].axis('off')
        
        #Ocultar ejes sobrantes en la segunda fila
        for i in range(len(fn_examples), axes.shape[1]):
            axes[1, i].set_visible(False)
        
        plt.suptitle('Análisis de Errores', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'plots/{model_name}_error_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _analyze_error_confidence(self, false_positives, false_negatives, y_proba):
        """Analizar la confianza en las predicciones erróneas"""
        
        if len(false_positives) > 0:
            fp_confidence = y_proba[false_positives]
            print(f"\n Análisis de confianza - Falsos Positivos:")
            print(f"   Confianza promedio: {fp_confidence.mean():.3f}")
            print(f"   Confianza mínima: {fp_confidence.min():.3f}")
            print(f"   Confianza máxima: {fp_confidence.max():.3f}")
        
        if len(false_negatives) > 0:
            fn_confidence = 1 - y_proba[false_negatives]  # Confianza en clase negativa
            print(f"\n Análisis de confianza - Falsos Negativos:")
            print(f"   Confianza promedio: {fn_confidence.mean():.3f}")
            print(f"   Confianza mínima: {fn_confidence.min():.3f}")
            print(f"   Confianza máxima: {fn_confidence.max():.3f}")
    
    def create_results_summary(self):
        """Crear resumen comparativo de todos los modelos"""
        print(f"\n RESUMEN COMPARATIVO DE MODELOS")
        print("=" * 80)
        
        #Crear DataFrame con métricas principales
        summary_data = []
        for model_name, results in self.results.items():
            row = {
                'Modelo': model_name,
                'Accuracy': results['accuracy'],
                'F1-Score': results['f1_score'],
                'ROC AUC': results['roc_auc'],
                'PR AUC': results['pr_auc'],
                'Precision (Normal)': results['classification_report']['Normal']['precision'],
                'Recall (Normal)': results['classification_report']['Normal']['recall'],
                'Precision (Neumonía)': results['classification_report']['Neumonía']['precision'],
                'Recall (Neumonía)': results['classification_report']['Neumonía']['recall']
            }
            summary_data.append(row)
        
        df_summary = pd.DataFrame(summary_data)
        df_summary = df_summary.round(4)
        
        print(df_summary.to_string(index=False))
        
        #Guardar resultados
        df_summary.to_csv('results/model_comparison.csv', index=False)
        print(f"\n Resultados guardados en 'results/model_comparison.csv'")
        
        #Encontrar mejor modelo
        best_f1_idx = df_summary['F1-Score'].idxmax()
        best_auc_idx = df_summary['ROC AUC'].idxmax()
        
        print(f"\n MEJORES MODELOS:")
        print(f"   Mejor F1-Score: {df_summary.loc[best_f1_idx, 'Modelo']} ({df_summary.loc[best_f1_idx, 'F1-Score']:.4f})")
        print(f"   Mejor ROC AUC: {df_summary.loc[best_auc_idx, 'Modelo']} ({df_summary.loc[best_auc_idx, 'ROC AUC']:.4f})")
        
        return df_summary
    
    def clinical_impact_analysis(self, model_name):
        """Análisis del impacto clínico del modelo"""
        if model_name not in self.results:
            print(f" No hay resultados para el modelo '{model_name}'")
            return
        
        results = self.results[model_name]
        cm = results['confusion_matrix']
        
        #Extraer valores de la matriz de confusión
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\n ANÁLISIS DE IMPACTO CLÍNICO - {model_name}")
        print("=" * 60)
        print(f" Matriz de Confusión:")
        print(f"   Verdaderos Negativos (TN): {tn} - Normales correctamente identificados")
        print(f"   Falsos Positivos (FP): {fp} - Normales diagnosticados como neumonía")
        print(f"   Falsos Negativos (FN): {fn} - Neumonías no detectadas")
        print(f"   Verdaderos Positivos (TP): {tp} - Neumonías correctamente detectadas")
        
        #Calcular métricas clínicas
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  #Valor Predictivo Positivo
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  #Valor Predictivo Negativo
        
        print(f"\n Métricas Clínicas:")
        print(f"   Sensibilidad (Recall): {sensitivity:.4f} - Capacidad de detectar neumonía")
        print(f"   Especificidad: {specificity:.4f} - Capacidad de identificar casos normales")
        print(f"   Valor Predictivo Positivo: {ppv:.4f} - Probabilidad de neumonía si predicción es positiva")
        print(f"   Valor Predictivo Negativo: {npv:.4f} - Probabilidad de normalidad si predicción es negativa")
        
        # Interpretación clínica
        print(f"\n Interpretación Clínica:")
        if sensitivity >= 0.9:
            print("    EXCELENTE sensibilidad - Muy pocas neumonías quedarán sin detectar")
        elif sensitivity >= 0.8:
            print("    BUENA sensibilidad - Detecta la mayoría de casos de neumonía")
        else:
            print("    BAJA sensibilidad - Riesgo de no detectar neumonías (falsos negativos)")
        
        if specificity >= 0.9:
            print("    EXCELENTE especificidad - Muy pocos falsos positivos")
        elif specificity >= 0.8:
            print("    BUENA especificidad - Pocos casos normales mal clasificados")
        else:
            print("    BAJA especificidad - Muchos falsos positivos (sobrediagnóstico)")
        
        # Recomendaciones
        print(f"\n Recomendaciones:")
        if fn > fp:
            print("   - Priorizar la reducción de falsos negativos (casos de neumonía no detectados)")
            print("   - Considerar ajustar el umbral de decisión hacia valores más bajos")
        elif fp > fn:
            print("   - Reducir falsos positivos para evitar sobrediagnóstico")
            print("   - Considerar ajustar el umbral de decisión hacia valores más altos")
        else:
            print("   - Balance adecuado entre falsos positivos y negativos")
        
        return {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
        }

    #NUEVOS MÉTODOS GRAD-CAM
    
    def _find_target_layer(self, model):
        """Encontrar la capa objetivo para Grad-CAM"""
        
        #Para modelos con transfer learning, buscar en la base model
        for layer in reversed(model.layers):
            if hasattr(layer, 'layers'):  #Es un modelo anidado (transfer learning)
                for sublayer in reversed(layer.layers):
                    if len(sublayer.output_shape) == 4:  #Capa convolucional
                        return sublayer.name
            elif len(layer.output_shape) == 4:  #Capa convolucional directa
                return layer.name
        
        return None
    
    def generate_gradcam_heatmap(self, model, img_array, pred_index=None):
        """
        Generar heatmap de Grad-CAM
        
        Args:
            model: Modelo de Keras
            img_array: Array de imagen (1, 224, 224, 3)
            pred_index: Índice de clase (None para usar predicción)
        
        Returns:
            heatmap: Array numpy normalizado del heatmap
        """
        
        #Encontrar capa objetivo
        last_conv_layer_name = self._find_target_layer(model)
        
        if last_conv_layer_name is None:
            print(" No se pudo encontrar capa convolucional")
            return None
        
        print(f" Usando capa: {last_conv_layer_name}")
        
        #Crear modelo que mapea entrada a activaciones y predicciones
        grad_model = keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        #Calcular gradientes de la predicción con respecto a las activaciones
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        
        #Gradientes de la clase predicha con respecto a las activaciones
        grads = tape.gradient(class_channel, last_conv_layer_output)
        
        #Vector de importancia promedio por canal
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        #Multiplicar cada canal por su importancia y sumar
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        #Normalizar heatmap entre 0 y 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    
    def create_gradcam_overlay(self, img, heatmap, alpha=0.4):
        """
        Crear overlay de Grad-CAM sobre imagen original
        
        Args:
            img: Imagen original (224, 224, 3) en rango [0, 1]
            heatmap: Heatmap de Grad-CAM
            alpha: Transparencia del overlay
        
        Returns:
            superimposed_img: Imagen con overlay
        """
        
        #Redimensionar heatmap al tamaño de la imagen
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        #Convertir heatmap a colormap
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        #Convertir img a formato uint8 si está normalizada
        if img.max() <= 1.0:
            img = np.uint8(255 * img)
        
        #Asegurar que ambas imágenes estén en el mismo formato
        heatmap = heatmap.astype(np.float32)
        img = img.astype(np.float32)
        
        #Crear superposición
        superimposed_img = heatmap * alpha + img * (1 - alpha)
        superimposed_img = np.uint8(superimposed_img)
        
        return superimposed_img
    
    def analyze_with_gradcam(self, model, model_name, test_generator, n_examples=6):
        """
        Análisis con Grad-CAM para casos específicos
        
        Args:
            model: Modelo de Keras entrenado
            model_name: Nombre del modelo
            test_generator: Generador de datos de prueba
            n_examples: Número de ejemplos a analizar
        """
        
        print(f"\n ANÁLISIS GRAD-CAM - {model_name}")
        print("=" * 60)
        
        if model_name not in self.results:
            print(f" Primero evalúa el modelo {model_name}")
            return None
        
        #Obtener datos de resultados
        y_true = self.results[model_name]['true_labels']
        y_pred = self.results[model_name]['predictions']
        y_proba = self.results[model_name]['probabilities']
        
        #Encontrar casos interesantes para análisis
        true_positives = np.where((y_true == 1) & (y_pred == 1))[0]    #Neumonías bien detectadas
        true_negatives = np.where((y_true == 0) & (y_pred == 0))[0]    #Normales bien detectadas
        false_positives = np.where((y_true == 0) & (y_pred == 1))[0]   #Falsos positivos
        false_negatives = np.where((y_true == 1) & (y_pred == 0))[0]   # Falsos negativos
        
        print(f" Casos disponibles:")
        print(f"   Verdaderos Positivos: {len(true_positives)}")
        print(f"   Verdaderos Negativos: {len(true_negatives)}")
        print(f"   Falsos Positivos: {len(false_positives)}")
        print(f"   Falsos Negativos: {len(false_negatives)}")
        
        #Seleccionar ejemplos balanceados
        selected_indices = []
        case_labels = []
        
        #2 casos correctos de cada clase
        if len(true_positives) > 0:
            selected = np.random.choice(true_positives, min(2, len(true_positives)), replace=False)
            selected_indices.extend(selected)
            case_labels.extend(['✅ Neumonía Correcta'] * len(selected))
        
        if len(true_negatives) > 0:
            selected = np.random.choice(true_negatives, min(2, len(true_negatives)), replace=False)
            selected_indices.extend(selected)
            case_labels.extend(['✅ Normal Correcto'] * len(selected))
        
        #1 error de cada tipo si existen
        if len(false_positives) > 0:
            selected = np.random.choice(false_positives, min(1, len(false_positives)), replace=False)
            selected_indices.extend(selected)
            case_labels.extend([' Falso Positivo'] * len(selected))
        
        if len(false_negatives) > 0:
            selected = np.random.choice(false_negatives, min(1, len(false_negatives)), replace=False)
            selected_indices.extend(selected)
            case_labels.extend([' Falso Negativo'] * len(selected))
        
        n_examples = min(len(selected_indices), 6)
        
        if n_examples == 0:
            print(" No hay casos para analizar")
            return None
        
        #Obtener imágenes del generador
        print(" Extrayendo imágenes del generador...")
        test_generator.reset()
        all_images = []
        
        for i in range(len(test_generator)):
            batch_images, _ = test_generator[i]
            all_images.extend(batch_images)
            if len(all_images) >= test_generator.samples:
                break
        
        all_images = np.array(all_images[:test_generator.samples])
        
        #Crear visualización
        fig, axes = plt.subplots(3, n_examples, figsize=(4*n_examples, 12))
        if n_examples == 1:
            axes = axes.reshape(3, 1)
        
        print(" Generando visualizaciones Grad-CAM...")
        
        for i in range(n_examples):
            idx = selected_indices[i]
            case_label = case_labels[i]
            
            #Obtener imagen
            img = all_images[idx]
            img_array = np.expand_dims(img, axis=0)
            
            #Hacer predicción
            prediction = model.predict(img_array, verbose=0)[0][0]
            predicted_class = "Neumonía" if prediction > 0.5 else "Normal"
            true_class = "Neumonía" if y_true[idx] == 1 else "Normal"
            
            #Generar Grad-CAM
            heatmap = self.generate_gradcam_heatmap(model, img_array)
            
            #Mostrar imagen original
            axes[0, i].imshow(img)
            axes[0, i].set_title(f'Original\n{case_label}')
            axes[0, i].axis('off')
            
            #Mostrar heatmap
            if heatmap is not None:
                im = axes[1, i].imshow(heatmap, cmap='jet')
                axes[1, i].set_title(f'Grad-CAM Heatmap')
                axes[1, i].axis('off')
                
                #Mostrar overlay
                overlay = self.create_gradcam_overlay(img, heatmap)
                axes[2, i].imshow(overlay)
                axes[2, i].set_title(f'Overlay\nVerdad: {true_class}\nPredicción: {predicted_class}\nConfianza: {prediction:.3f}')
                axes[2, i].axis('off')
            else:
                axes[1, i].text(0.5, 0.5, 'Error\nGrad-CAM', ha='center', va='center')
                axes[2, i].text(0.5, 0.5, 'Error\nGrad-CAM', ha='center', va='center')
                axes[1, i].axis('off')
                axes[2, i].axis('off')
        
        plt.suptitle(f'Análisis Grad-CAM - {model_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        #Guardar visualización
        save_path = f'plots/{model_name}_gradcam_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f" Análisis guardado en: {save_path}")
        
        #Interpretación automática
        self._interpret_gradcam_results(model_name, n_examples)
        
        return selected_indices, case_labels
    
    def analyze_gradcam_for_errors_only(self, model, model_name, test_generator, max_errors=8):
        """
        Análisis Grad-CAM específicamente para casos de error
        
        Args:
            model: Modelo de Keras
            model_name: Nombre del modelo
            test_generator: Generador de prueba
            max_errors: Máximo número de errores a analizar
        """
        
        print(f"\n ANÁLISIS GRAD-CAM DE ERRORES - {model_name}")
        print("=" * 60)
        
        if model_name not in self.results:
            print(f" Primero evalúa el modelo {model_name}")
            return None
        
        #Obtener casos de error
        y_true = self.results[model_name]['true_labels']
        y_pred = self.results[model_name]['predictions']
        y_proba = self.results[model_name]['probabilities']
        
        false_positives = np.where((y_true == 0) & (y_pred == 1))[0]
        false_negatives = np.where((y_true == 1) & (y_pred == 0))[0]
        
        print(f"   Errores encontrados:")
        print(f"   Falsos Positivos: {len(false_positives)}")
        print(f"   Falsos Negativos: {len(false_negatives)}")
        
        if len(false_positives) == 0 and len(false_negatives) == 0:
            print(" ¡No hay errores para analizar! Modelo perfecto.")
            return None
        
        #Seleccionar errores para análisis
        error_indices = []
        error_types = []
        
        #Tomar muestra de falsos positivos
        if len(false_positives) > 0:
            n_fp = min(max_errors // 2, len(false_positives))
            selected_fp = np.random.choice(false_positives, n_fp, replace=False)
            error_indices.extend(selected_fp)
            error_types.extend(['Falso Positivo'] * n_fp)
        
        #Tomar muestra de falsos negativos
        if len(false_negatives) > 0:
            n_fn = min(max_errors // 2, len(false_negatives))
            selected_fn = np.random.choice(false_negatives, n_fn, replace=False)
            error_indices.extend(selected_fn)
            error_types.extend(['Falso Negativo'] * n_fn)
        
        n_errors = len(error_indices)
        
        #Obtener imágenes
        test_generator.reset()
        all_images = []
        for i in range(len(test_generator)):
            batch_images, _ = test_generator[i]
            all_images.extend(batch_images)
            if len(all_images) >= test_generator.samples:
                break
        all_images = np.array(all_images[:test_generator.samples])
        
        #Crear visualización de errores
        cols = min(4, n_errors)
        rows = (n_errors + cols - 1) // cols
        
        fig, axes = plt.subplots(rows * 2, cols, figsize=(5*cols, 6*rows))
        if rows == 1 and cols == 1:
            axes = axes.reshape(2, 1)
        elif rows == 1:
            axes = axes.reshape(2, cols)
        
        for i in range(n_errors):
            idx = error_indices[i]
            error_type = error_types[i]
            
            row = (i // cols) * 2
            col = i % cols
            
            #Obtener imagen y datos
            img = all_images[idx]
            img_array = np.expand_dims(img, axis=0)
            
            prediction = y_proba[idx]
            true_class = "Neumonía" if y_true[idx] == 1 else "Normal"
            pred_class = "Neumonía" if prediction > 0.5 else "Normal"
            
            #Generar Grad-CAM
            heatmap = self.generate_gradcam_heatmap(model, img_array)
            
            #Imagen original
            axes[row, col].imshow(img)
            axes[row, col].set_title(f'{error_type}\nVerdad: {true_class}\nPredicción: {pred_class}')
            axes[row, col].axis('off')
            
            #Overlay con Grad-CAM
            if heatmap is not None:
                overlay = self.create_gradcam_overlay(img, heatmap)
                axes[row + 1, col].imshow(overlay)
                axes[row + 1, col].set_title(f'Grad-CAM (Conf: {prediction:.3f})')
                axes[row + 1, col].axis('off')
            else:
                axes[row + 1, col].text(0.5, 0.5, 'Error Grad-CAM', ha='center', va='center')
                axes[row + 1, col].axis('off')
        
        #Ocultar axes sobrantes
        total_subplots = rows * 2 * cols
        for i in range(n_errors, total_subplots // 2):
            row = (i // cols) * 2
            col = i % cols
            if row < len(axes) and col < len(axes[0]):
                axes[row, col].set_visible(False)
                axes[row + 1, col].set_visible(False)
        
        plt.suptitle(f'Análisis Grad-CAM de Errores - {model_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        #Guardar
        save_path = f'plots/{model_name}_gradcam_errors.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f" Análisis de errores guardado en: {save_path}")
        
        #Análisis de errores
        self._analyze_error_patterns(error_types, model_name)
        
        return error_indices, error_types
    
    def _interpret_gradcam_results(self, model_name, n_cases):
        """Proporcionar interpretación automática de los resultados Grad-CAM"""
        
        print(f"\n INTERPRETACIÓN GRAD-CAM PARA {model_name}:")
        print("=" * 50)
        print(" Cómo interpretar los resultados:")
        print()
        print(" HEATMAP (imagen del medio):")
        print("    Rojo/Amarillo: Áreas de ALTA importancia para la decisión")
        print("    Azul/Verde: Áreas de BAJA importancia")
        print()
        print(" PARA CASOS DE NEUMONÍA:")
        print("    BUENO: Activación en áreas de consolidación/infiltrado")
        print("    BUENO: Foco en regiones pulmonares anómalas")
        print("    MALO: Activación en bordes de la imagen o artefactos")
        print()
        print(" PARA CASOS NORMALES:")
        print("    BUENO: Activación distribuida o en estructuras anatómicas")
        print("    BUENO: No foco excesivo en una región específica")
        print("    MALO: Activación intensa en regiones que parecen normales")
        print()
        print(" ANÁLISIS DE ERRORES:")
        print("    Falsos Positivos: ¿El modelo ve 'patrones' donde no los hay?")
        print("    Falsos Negativos: ¿El modelo ignora regiones obviamente anómalas?")
        print()
        print(f" Se analizaron {n_cases} casos representativos del modelo {model_name}")
    
    def _analyze_error_patterns(self, error_types, model_name):
        """Analizar patrones en los errores"""
        
        fp_count = error_types.count('Falso Positivo')
        fn_count = error_types.count('Falso Negativo')
        
        print(f"\n ANÁLISIS DE PATRONES DE ERROR - {model_name}:")
        print("=" * 50)
        print(f" Distribución de errores analizados:")
        print(f"   Falsos Positivos: {fp_count}")
        print(f"   Falsos Negativos: {fn_count}")
        print()
        
        if fp_count > fn_count:
            print("  PATRÓN: Más falsos positivos que falsos negativos")
            print(" Posibles causas:")
            print("   - Modelo muy sensible, detecta patrones inexistentes")
            print("   - Sobreentrenamiento en características de neumonía")
            print("   - Necesita ajustar umbral de decisión (aumentar)")
        elif fn_count > fp_count:
            print("  PATRÓN: Más falsos negativos que falsos positivos")
            print("  Posibles causas:")
            print("   - Modelo muy conservador, pierde casos reales")
            print("   - Insuficiente entrenamiento en variaciones de neumonía")
            print("   - Necesita ajustar umbral de decisión (disminuir)")
        else:
            print(" PATRÓN: Balance entre tipos de error")
            print(" El modelo tiene un comportamiento balanceado")
        
        print()
        print(" Qué buscar en los Grad-CAM de errores:")
        print("    Falsos Positivos: ¿Se enfoca en artefactos o bordes?")
        print("    Falsos Negativos: ¿Ignora áreas obviamente anómalas?")
        print("    ¿Hay patrones consistentes en los errores?")
    
    def complete_analysis_with_gradcam(self, model, model_name, test_generator):
        """
        Análisis completo que incluye Grad-CAM automáticamente
        
        Esta función ejecuta:
        1. Evaluación estándar
        2. Análisis Grad-CAM general
        3. Análisis Grad-CAM de errores
        4. Interpretación clínica
        """
        
        print(f"\n ANÁLISIS COMPLETO CON GRAD-CAM - {model_name}")
        print("=" * 70)
        
        #1. Evaluación estándar
        results = self.evaluate_model(model, test_generator, model_name)
        
        #2. Análisis Grad-CAM general
        print(f"\n Fase 1: Análisis Grad-CAM General")
        self.analyze_with_gradcam(model, model_name, test_generator, n_examples=6)
        
        #3. Análisis Grad-CAM de errores específicos
        print(f"\n Fase 2: Análisis Grad-CAM de Errores")
        self.analyze_gradcam_for_errors_only(model, model_name, test_generator, max_errors=6)
        
        #4. Análisis clínico
        print(f"\n Fase 3: Análisis de Impacto Clínico")
        clinical_results = self.clinical_impact_analysis(model_name)
        
        print(f"\n ANÁLISIS COMPLETO FINALIZADO PARA {model_name}")
        print(" Archivos generados:")
        print(f"   plots/{model_name}_gradcam_analysis.png")
        print(f"   plots/{model_name}_gradcam_errors.png")
        
        return results, clinical_results


#FUNCIÓN PRINCIPAL ACTUALIZADA

def complete_evaluation(model_manager, test_generator, config):
    """Realizar evaluación completa de todos los modelos CON GRAD-CAM"""
    print(" INICIANDO EVALUACIÓN COMPLETA DE MODELOS CON GRAD-CAM")
    print("=" * 60)
    
    evaluator = PneumoniaEvaluator(config)
    
    #Evaluar todos los modelos disponibles CON GRAD-CAM
    for model_name, model in model_manager.models.items():
        if 'best' not in model_name:  #Solo evaluar modelos originales
            print(f"\n{'='*70}")
            print(f" EVALUANDO: {model_name.upper()}")
            print(f"{'='*70}")
            
            #Análisis completo (evaluación + Grad-CAM)
            evaluator.complete_analysis_with_gradcam(model, model_name, test_generator)
    
    #Crear visualizaciones comparativas (tus funciones originales)
    print(f"\n Generando visualizaciones comparativas...")
    evaluator.plot_confusion_matrices()
    evaluator.plot_roc_curves()
    evaluator.plot_precision_recall_curves()
    
    #Crear resumen de resultados
    summary_df = evaluator.create_results_summary()
    
    #Análisis detallado del mejor modelo
    best_model = summary_df.loc[summary_df['F1-Score'].idxmax(), 'Modelo']
    print(f"\n Mejor modelo identificado: {best_model}")
    
    print(f"\n EVALUACIÓN COMPLETA CON GRAD-CAM FINALIZADA")
    print(" Archivos generados:")
    print("    Matrices de confusión, curvas ROC, etc.")
    print("    Análisis Grad-CAM para todos los modelos")
    print("    Análisis Grad-CAM de errores específicos")
    print("    Resumen comparativo completo")
    
    return evaluator, summary_df


#  FUNCIÓN PARA USAR SOLO GRAD-CAM

def analyze_single_model_with_gradcam(model, model_name, test_generator, config):
    """
    Función para analizar un solo modelo con Grad-CAM
    
    Útil si solo quieres probar Grad-CAM en tu mejor modelo
    """
    
    print(f" ANÁLISIS GRAD-CAM INDIVIDUAL - {model_name}")
    print("=" * 50)

    evaluator = PneumoniaEvaluator(config)
    # Evaluar el modelo
    results = evaluator.evaluate_model(model, test_generator, model_name)
    
    #Análisis Grad-CAM
    evaluator.analyze_with_gradcam(model, model_name, test_generator)
    #Análisis de errores con Grad-CAM
    evaluator.analyze_gradcam_for_errors_only(model, model_name, test_generator)
    #Análisis clínico
    clinical_results = evaluator.clinical_impact_analysis(model_name)
    print(f"Análisis individual completado para {model_name}")
    return evaluator, results, clinical_results


# Ejemplo de uso
if __name__ == "__main__":
    # Este código se ejecutaría después de entrenar los modelos
    print("Ejemplos de uso:")
    print("1. Evaluación completa: evaluator, summary = complete_evaluation(model_manager, test_gen, config)")
    print("2. Un solo modelo: evaluator, results, clinical = analyze_single_model_with_gradcam(model, 'resnet50', test_gen, config)")