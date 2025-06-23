#Evaluación y Análisis de Resultados - Detección de Neumonía
#Métricas, visualizaciones y análisis de errores

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
    """Clase para evaluar modelos de detección de neumonía"""
    
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
        
        #calcular métricas
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        #report detallado
        report = classification_report(y_true, y_pred, 
                                     target_names=['Normal', 'Neumonía'], 
                                     output_dict=True)
        
        #matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        
        #ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        #precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        #guardar resultados
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
        
        #mostrar métricas principales
        print(f"✅ Resultados para {model_name}:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   ROC AUC: {roc_auc:.4f}")
        print(f"   PR AUC: {pr_auc:.4f}")
        
        #métricas por clase
        print(f"\n Métricas por clase:")
        for class_name, metrics in report.items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                print(f"   {class_name}:")
                print(f"     Precision: {metrics['precision']:.4f}")
                print(f"     Recall: {metrics['recall']:.4f}")
                print(f"     F1-Score: {metrics['f1-score']:.4f}")
        
        return self.results[model_name]
    
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
            
            #normalizar matriz de confusión
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            #crear heatmap
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=['Normal', 'Neumonía'],
                       yticklabels=['Normal', 'Neumonía'],
                       ax=axes[i])
            
            axes[i].set_title(f'{model_name}\nAccuracy: {self.results[model_name]["accuracy"]:.3f}')
            axes[i].set_xlabel('Predicción')
            axes[i].set_ylabel('Verdad')
        
        #ocultar ejes sobrantes
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
        
        #línea diagonal (clasificador aleatorio)
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
        
        #obtener datos
        y_true = self.results[model_name]['true_labels']
        y_pred = self.results[model_name]['predictions']
        y_proba = self.results[model_name]['probabilities']
        
        #identificar errores
        false_positives = np.where((y_true == 0) & (y_pred == 1))[0]
        false_negatives = np.where((y_true == 1) & (y_pred == 0))[0]
        
        print(f" Resumen de errores:")
        print(f"   Falsos Positivos: {len(false_positives)}")
        print(f"   Falsos Negativos: {len(false_negatives)}")
        print(f"   Total de errores: {len(false_positives) + len(false_negatives)}")
        
        #mostrar ejemplos de errores
        self._show_error_examples(test_generator, false_positives, false_negatives, 
                                y_proba, n_examples)
        
        #análisis de confianza en errores
        self._analyze_error_confidence(false_positives, false_negatives, y_proba)
        
        return false_positives, false_negatives
    
    def _show_error_examples(self, test_generator, false_positives, false_negatives, 
                           y_proba, n_examples):
        """Mostrar ejemplos de errores"""
        
        # Seleccionar ejemplos aleatorios
        fp_examples = np.random.choice(false_positives, 
                                     min(n_examples//2, len(false_positives)), 
                                     replace=False) if len(false_positives) > 0 else []
        fn_examples = np.random.choice(false_negatives, 
                                     min(n_examples//2, len(false_negatives)), 
                                     replace=False) if len(false_negatives) > 0 else []
        
        fig, axes = plt.subplots(2, max(len(fp_examples), len(fn_examples)), 
                               figsize=(4*max(len(fp_examples), len(fn_examples)), 8))
        
        if len(fp_examples) == 0 and len(fn_examples) == 0:
            print("✅ ¡No hay errores para mostrar!")
            return
        
        #obtener imágenes del generador
        test_generator.reset()
        images = []
        for i in range(len(test_generator)):
            batch_images, _ = test_generator[i]
            images.extend(batch_images)
            if len(images) >= test_generator.samples:
                break
        images = np.array(images[:test_generator.samples])
        
        #mostrar falsos positivos
        for i, idx in enumerate(fp_examples):
            if i < axes.shape[1]:
                axes[0, i].imshow(images[idx])
                axes[0, i].set_title(f'Falso Positivo\nConfianza: {y_proba[idx]:.3f}')
                axes[0, i].axis('off')
        
        #ccultar ejes sobrantes en la primera fila
        for i in range(len(fp_examples), axes.shape[1]):
            axes[0, i].set_visible(False)
        
        #mostrar falsos negativos
        for i, idx in enumerate(fn_examples):
            if i < axes.shape[1]:
                axes[1, i].imshow(images[idx])
                axes[1, i].set_title(f'Falso Negativo\nConfianza: {y_proba[idx]:.3f}')
                axes[1, i].axis('off')
        
        #ccultar ejes sobrantes en la segunda fila
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
            print(f"\n📈 Análisis de confianza - Falsos Positivos:")
            print(f"   Confianza promedio: {fp_confidence.mean():.3f}")
            print(f"   Confianza mínima: {fp_confidence.min():.3f}")
            print(f"   Confianza máxima: {fp_confidence.max():.3f}")
        
        if len(false_negatives) > 0:
            fn_confidence = 1 - y_proba[false_negatives]  # Confianza en clase negativa
            print(f"\n📈 Análisis de confianza - Falsos Negativos:")
            print(f"   Confianza promedio: {fn_confidence.mean():.3f}")
            print(f"   Confianza mínima: {fn_confidence.min():.3f}")
            print(f"   Confianza máxima: {fn_confidence.max():.3f}")
    
    def create_results_summary(self):
        """Crear resumen comparativo de todos los modelos"""
        print(f"\n RESUMEN COMPARATIVO DE MODELOS")
        print("=" * 80)
        
        #crear DataFrame con métricas principales
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
        
        # Guardar resultados
        df_summary.to_csv('results/model_comparison.csv', index=False)
        print(f"\n Resultados guardados en 'results/model_comparison.csv'")
        
        # Encontrar mejor modelo
        best_f1_idx = df_summary['F1-Score'].idxmax()
        best_auc_idx = df_summary['ROC AUC'].idxmax()
        
        print(f"\n MEJORES MODELOS:")
        print(f"   Mejor F1-Score: {df_summary.loc[best_f1_idx, 'Modelo']} ({df_summary.loc[best_f1_idx, 'F1-Score']:.4f})")
        print(f"   Mejor ROC AUC: {df_summary.loc[best_auc_idx, 'Modelo']} ({df_summary.loc[best_auc_idx, 'ROC AUC']:.4f})")
        
        return df_summary
    
    def clinical_impact_analysis(self, model_name):
        """Análisis del impacto clínico del modelo"""
        if model_name not in self.results:
            print(f"❌ No hay resultados para el modelo '{model_name}'")
            return
        
        results = self.results[model_name]
        cm = results['confusion_matrix']
        
        #extraer valores de la matriz de confusión
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\n ANÁLISIS DE IMPACTO CLÍNICO - {model_name}")
        print("=" * 60)
        print(f"   Matriz de Confusión:")
        print(f"   Verdaderos Negativos (TN): {tn} - Normales correctamente identificados")
        print(f"   Falsos Positivos (FP): {fp} - Normales diagnosticados como neumonía")
        print(f"   Falsos Negativos (FN): {fn} - Neumonías no detectadas")
        print(f"   Verdaderos Positivos (TP): {tp} - Neumonías correctamente detectadas")
        
        #calcular métricas clínicas
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  #valor Predictivo Positivo
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  #valor Predictivo Negativo
        
        print(f"\n Métricas Clínicas:")
        print(f"   Sensibilidad (Recall): {sensitivity:.4f} - Capacidad de detectar neumonía")
        print(f"   Especificidad: {specificity:.4f} - Capacidad de identificar casos normales")
        print(f"   Valor Predictivo Positivo: {ppv:.4f} - Probabilidad de neumonía si predicción es positiva")
        print(f"   Valor Predictivo Negativo: {npv:.4f} - Probabilidad de normalidad si predicción es negativa")
        
        #Interpretación clínica
        print(f"\n Interpretación Clínica:")
        if sensitivity >= 0.9:
            print("    EXCELENTE sensibilidad - Muy pocas neumonías quedarán sin detectar")
        elif sensitivity >= 0.8:
            print("    BUENA sensibilidad - Detecta la mayoría de casos de neumonía")
        else:
            print("     BAJA sensibilidad - Riesgo de no detectar neumonías (falsos negativos)")
        
        if specificity >= 0.9:
            print("    EXCELENTE especificidad - Muy pocos falsos positivos")
        elif specificity >= 0.8:
            print("   BUENA especificidad - Pocos casos normales mal clasificados")
        else:
            print("    BAJA especificidad - Muchos falsos positivos (sobrediagnóstico)")
        
        #recomendaciones
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

#función principal para evaluación completa
def complete_evaluation(model_manager, test_generator, config):
    """Realizar evaluación completa de todos los modelos"""
    print("INICIANDO EVALUACIÓN COMPLETA DE MODELOS")
    print("=" * 60)
    
    evaluator = PneumoniaEvaluator(config)
    
    #Evaluar todos los modelos disponibles
    for model_name, model in model_manager.models.items():
        if 'best' not in model_name:  # Solo evaluar modelos originales
            evaluator.evaluate_model(model, test_generator, model_name)
    
    #crear visualizaciones comparativas
    print(f"\n Generando visualizaciones comparativas...")
    evaluator.plot_confusion_matrices()
    evaluator.plot_roc_curves()
    evaluator.plot_precision_recall_curves()
    
    # Crear resumen de resultados
    summary_df = evaluator.create_results_summary()
    
    #análisis detallado del mejor modelo
    best_model = summary_df.loc[summary_df['F1-Score'].idxmax(), 'Modelo']
    print(f"\n Análisis detallado del mejor modelo: {best_model}")
    
    #análisis de errores
    evaluator.analyze_errors(best_model, test_generator)
    
    #análisis de impacto clínico
    evaluator.clinical_impact_analysis(best_model)
    
    return evaluator, summary_df

#ejemplo de uso
if __name__ == "__main__":
    #este código se ejecutaría después de entrenar los modelos
    print("Ejemplo de evaluación completa")
    print("Ejecutar después de completar el entrenamiento de modelos")