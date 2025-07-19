import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from collections import Counter
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class PneumoScanEDA:
    """Análisis Exploratorio Multi-Dimensional para PneumoScan AI"""
    
    def __init__(self, data_path="data/chest_xray"):
        self.data_path = Path(data_path)
        self.results_path = Path("results/cpu_local/eda")
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        # Configurar estilo de plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def analyze_dataset_structure(self):
        """FASE 1: Análisis de estructura del dataset"""
        print(" ANÁLISIS DE ESTRUCTURA DEL DATASET")
        print("=" * 50)
        
        dataset_info = {}
        
        for split in ['train', 'val', 'test']:
            split_path = self.data_path / split
            if not split_path.exists():
                continue
                
            split_info = {}
            total_images = 0
            
            for class_name in ['NORMAL', 'PNEUMONIA']:
                class_path = split_path / class_name
                if class_path.exists():
                    images = list(class_path.glob('*.jpeg')) + list(class_path.glob('*.jpg'))
                    split_info[class_name] = len(images)
                    total_images += len(images)
            
            split_info['TOTAL'] = total_images
            dataset_info[split] = split_info
            
            print(f"\n{split.upper()}:")
            print(f"  • Normal: {split_info.get('NORMAL', 0):,} imágenes")
            print(f"  • Neumonía: {split_info.get('PNEUMONIA', 0):,} imágenes")
            print(f"  • Total: {total_images:,} imágenes")
            
            if split_info.get('NORMAL', 0) > 0 and split_info.get('PNEUMONIA', 0) > 0:
                ratio = split_info['PNEUMONIA'] / split_info['NORMAL']
                print(f"  • Ratio (Neumonía/Normal): {ratio:.2f}")
        
        # Visualización de distribución
        self._plot_class_distribution(dataset_info)
        
        return dataset_info
    
    def analyze_image_quality(self, sample_size=100):
        """FASE 2: Análisis de calidad de imagen"""
        print("\n ANÁLISIS DE CALIDAD DE IMAGEN")
        print("=" * 50)
        
        quality_metrics = {
            'brightness': [],
            'contrast': [],
            'sharpness': [],
            'noise_level': [],
            'resolution': [],
            'class': []
        }
        
        for split in ['train', 'test']:
            split_path = self.data_path / split
            if not split_path.exists():
                continue
                
            for class_name in ['NORMAL', 'PNEUMONIA']:
                class_path = split_path / class_name
                if not class_path.exists():
                    continue
                    
                images = list(class_path.glob('*.jpeg'))[:sample_size//4]
                
                for img_path in images:
                    try:
                        # Cargar imagen
                        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                        if img is None:
                            continue
                        
                        # Métricas de calidad
                        brightness = np.mean(img)
                        contrast = np.std(img)
                        
                        # Sharpness (Laplacian variance)
                        laplacian = cv2.Laplacian(img, cv2.CV_64F)
                        sharpness = laplacian.var()
                        
                        # Noise level (estimación)
                        noise = np.std(img - cv2.GaussianBlur(img, (5,5), 0))
                        
                        # Resolución
                        resolution = img.shape[0] * img.shape[1]
                        
                        quality_metrics['brightness'].append(brightness)
                        quality_metrics['contrast'].append(contrast)
                        quality_metrics['sharpness'].append(sharpness)
                        quality_metrics['noise_level'].append(noise)
                        quality_metrics['resolution'].append(resolution)
                        quality_metrics['class'].append(class_name)
                        
                    except Exception as e:
                        print(f"Error procesando {img_path}: {e}")
                        continue
        
        # Convertir a DataFrame
        df_quality = pd.DataFrame(quality_metrics)
        
        # Estadísticas por clase
        print("\n ESTADÍSTICAS DE CALIDAD POR CLASE:")
        for class_name in ['NORMAL', 'PNEUMONIA']:
            class_data = df_quality[df_quality['class'] == class_name]
            if len(class_data) > 0:
                print(f"\n{class_name}:")
                print(f"  • Brillo promedio: {class_data['brightness'].mean():.1f}")
                print(f"  • Contraste promedio: {class_data['contrast'].mean():.1f}")
                print(f"  • Nitidez promedio: {class_data['sharpness'].mean():.1f}")
                print(f"  • Nivel de ruido: {class_data['noise_level'].mean():.1f}")
        
        # Visualización
        self._plot_quality_analysis(df_quality)
        
        return df_quality
    
    def analyze_visual_patterns(self, sample_size=20):
        """FASE 3: Análisis de patrones visuales distintivos"""
        print("\n ANÁLISIS DE PATRONES VISUALES")
        print("=" * 50)
        
        # Crear mosaico de imágenes representativas
        fig, axes = plt.subplots(2, sample_size//2, figsize=(20, 8))
        fig.suptitle('Patrones Visuales Distintivos - PneumoScan AI', fontsize=16)
        
        row_idx = 0
        for class_name in ['NORMAL', 'PNEUMONIA']:
            class_path = self.data_path / 'train' / class_name
            if not class_path.exists():
                continue
                
            images = list(class_path.glob('*.jpeg'))[:sample_size//2]
            
            for col_idx, img_path in enumerate(images):
                try:
                    img = cv2.imread(str(img_path))
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    axes[row_idx, col_idx].imshow(img_rgb)
                    axes[row_idx, col_idx].set_title(f'{class_name}\n{img_path.name}', fontsize=8)
                    axes[row_idx, col_idx].axis('off')
                    
                except Exception as e:
                    axes[row_idx, col_idx].text(0.5, 0.5, f'Error: {img_path.name}', 
                                              ha='center', va='center', transform=axes[row_idx, col_idx].transAxes)
                    axes[row_idx, col_idx].axis('off')
            
            row_idx += 1
        
        plt.tight_layout()
        plt.savefig(self.results_path / 'visual_patterns.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(" Mosaico de patrones visuales guardado en: visual_patterns.png")
    
    def analyze_data_imbalance(self):
        """FASE 4: Análisis detallado de desequilibrio de datos"""
        print("\n ANÁLISIS DE DESEQUILIBRIO DE DATOS")
        print("=" * 50)
        
        imbalance_data = {}
        
        for split in ['train', 'val', 'test']:
            split_path = self.data_path / split
            if not split_path.exists():
                continue
                
            normal_count = len(list((split_path / 'NORMAL').glob('*.jpeg'))) if (split_path / 'NORMAL').exists() else 0
            pneumonia_count = len(list((split_path / 'PNEUMONIA').glob('*.jpeg'))) if (split_path / 'PNEUMONIA').exists() else 0
            
            total = normal_count + pneumonia_count
            
            if total > 0:
                imbalance_data[split] = {
                    'normal': normal_count,
                    'pneumonia': pneumonia_count,
                    'total': total,
                    'normal_pct': (normal_count / total) * 100,
                    'pneumonia_pct': (pneumonia_count / total) * 100,
                    'imbalance_ratio': pneumonia_count / normal_count if normal_count > 0 else 0
                }
        
        # Mostrar análisis
        for split, data in imbalance_data.items():
            print(f"\n{split.upper()}:")
            print(f"  • Normal: {data['normal']:,} ({data['normal_pct']:.1f}%)")
            print(f"  • Neumonía: {data['pneumonia']:,} ({data['pneumonia_pct']:.1f}%)")
            print(f"  • Ratio de desequilibrio: {data['imbalance_ratio']:.2f}")
            
            if data['imbalance_ratio'] > 2:
                print(f"    DESEQUILIBRIO SIGNIFICATIVO detectado")
            elif data['imbalance_ratio'] > 1.5:
                print(f"    Desequilibrio moderado")
            else:
                print(f"   Desequilibrio aceptable")
        
        # Visualización
        self._plot_imbalance_analysis(imbalance_data)
        
        return imbalance_data
    
    def generate_comprehensive_report(self):
        """FASE 5: Generar reporte completo de EDA"""
        print("\n GENERANDO REPORTE COMPLETO")
        print("=" * 50)
        
        # Ejecutar todos los análisis
        dataset_info = self.analyze_dataset_structure()
        quality_data = self.analyze_image_quality()
        self.analyze_visual_patterns()
        imbalance_data = self.analyze_data_imbalance()
        
        # Crear reporte textual
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                 REPORTE EDA - PNEUMOSCAN AI                  ║
╚══════════════════════════════════════════════════════════════╝

 RESUMEN EJECUTIVO:
─────────────────────
• Dataset analizado: {self.data_path}
• Análisis realizado: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

 DISTRIBUCIÓN DE DATOS:
─────────────────────────
"""
        
        total_images = 0
        for split, info in dataset_info.items():
            total_images += info.get('TOTAL', 0)
            report += f"• {split.upper()}: {info.get('TOTAL', 0):,} imágenes\n"
        
        report += f"• TOTAL GENERAL: {total_images:,} imágenes\n"
        
        if len(quality_data) > 0:
            report += f"""
 CALIDAD DE IMAGEN:
──────────────────────
• Brillo promedio: {quality_data['brightness'].mean():.1f}
• Contraste promedio: {quality_data['contrast'].mean():.1f}
• Nitidez promedio: {quality_data['sharpness'].mean():.1f}
• Nivel de ruido: {quality_data['noise_level'].mean():.1f}
"""
        
        report += f"""
 ANÁLISIS DE DESEQUILIBRIO:
──────────────────────────────
"""
        
        for split, data in imbalance_data.items():
            report += f"• {split.upper()}: Ratio {data['imbalance_ratio']:.2f} (Neumonía/Normal)\n"
        
        report += f"""
 RECOMENDACIONES:
─────────────────────
• Aplicar data augmentation para balancear clases
• Usar técnicas de weighted loss o focal loss
• Implementar validación cruzada estratificada
• Considerar métricas balanceadas (F1-score, AUC)
• Monitorear sensibilidad especialmente para casos críticos

 PRÓXIMOS PASOS:
─────────────────────
• Implementar estrategias de balanceo de datos
• Configurar métricas de evaluación médica
• Preparar pipeline de data augmentation
• Establecer benchmarks de performance
        """
        
        # Guardar reporte
        report_path = self.results_path / 'reporte_eda_completo.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        print(f"\n Reporte completo guardado en: {report_path}")
        
        return report
    
    def _plot_class_distribution(self, dataset_info):
        """Visualizar distribución de clases"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gráfico de barras
        splits = list(dataset_info.keys())
        normal_counts = [dataset_info[s].get('NORMAL', 0) for s in splits]
        pneumonia_counts = [dataset_info[s].get('PNEUMONIA', 0) for s in splits]
        
        x = np.arange(len(splits))
        width = 0.35
        
        axes[0].bar(x - width/2, normal_counts, width, label='Normal', alpha=0.8)
        axes[0].bar(x + width/2, pneumonia_counts, width, label='Neumonía', alpha=0.8)
        axes[0].set_xlabel('Split del Dataset')
        axes[0].set_ylabel('Número de Imágenes')
        axes[0].set_title('Distribución de Clases por Split')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(splits)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Gráfico de pie para entrenamiento
        if 'train' in dataset_info:
            train_data = dataset_info['train']
            sizes = [train_data.get('NORMAL', 0), train_data.get('PNEUMONIA', 0)]
            labels = ['Normal', 'Neumonía']
            colors = ['lightblue', 'lightcoral']
            
            axes[1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[1].set_title('Distribución de Clases - Entrenamiento')
        
        plt.tight_layout()
        plt.savefig(self.results_path / 'class_distribution.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def _plot_quality_analysis(self, df_quality):
        """Visualizar análisis de calidad"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Análisis de Calidad de Imagen - PneumoScan AI', fontsize=16)
        
        metrics = ['brightness', 'contrast', 'sharpness', 'noise_level']
        titles = ['Brillo', 'Contraste', 'Nitidez', 'Nivel de Ruido']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            row, col = i // 2, i % 2
            
            # Boxplot por clase
            df_quality.boxplot(column=metric, by='class', ax=axes[row, col])
            axes[row, col].set_title(f'{title} por Clase')
            axes[row, col].set_xlabel('Clase')
            axes[row, col].set_ylabel(title)
            
        plt.tight_layout()
        plt.savefig(self.results_path / 'quality_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def _plot_imbalance_analysis(self, imbalance_data):
        """Visualizar análisis de desequilibrio"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gráfico de barras de ratios
        splits = list(imbalance_data.keys())
        ratios = [imbalance_data[s]['imbalance_ratio'] for s in splits]
        
        bars = axes[0].bar(splits, ratios, color=['skyblue', 'lightgreen', 'lightcoral'])
        axes[0].set_xlabel('Split del Dataset')
        axes[0].set_ylabel('Ratio de Desequilibrio')
        axes[0].set_title('Ratio de Desequilibrio por Split\n(Neumonía/Normal)')
        axes[0].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Balance perfecto')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Agregar valores en las barras
        for bar, ratio in zip(bars, ratios):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{ratio:.2f}', ha='center', va='bottom')
        
        # Gráfico de líneas de porcentajes
        normal_pcts = [imbalance_data[s]['normal_pct'] for s in splits]
        pneumonia_pcts = [imbalance_data[s]['pneumonia_pct'] for s in splits]
        
        axes[1].plot(splits, normal_pcts, 'o-', label='Normal', linewidth=2, markersize=8)
        axes[1].plot(splits, pneumonia_pcts, 's-', label='Neumonía', linewidth=2, markersize=8)
        axes[1].set_xlabel('Split del Dataset')
        axes[1].set_ylabel('Porcentaje (%)')
        axes[1].set_title('Porcentaje de Clases por Split')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(self.results_path / 'imbalance_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()

def run_complete_eda():
    """Función principal para ejecutar EDA completo"""
    print(" PNEUMOSCAN AI - ANÁLISIS EXPLORATORIO DE DATOS")
    print("=" * 60)
    
    # Verificar que existe el dataset
    if not os.path.exists("data/chest_xray"):
        print(" ERROR: Dataset no encontrado en 'data/chest_xray'")
        print("   Coloca el dataset en la estructura correcta:")
        print("   data/chest_xray/")
        print("   ├── train/")
        print("   │   ├── NORMAL/")
        print("   │   └── PNEUMONIA/")
        print("   ├── val/")
        print("   └── test/")
        return
    
    # Crear instancia de EDA
    eda = PneumoScanEDA()
    
    # Ejecutar análisis completo
    try:
        report = eda.generate_comprehensive_report()
        
        print("\n ANÁLISIS COMPLETADO EXITOSAMENTE")
        print("=" * 60)
        print(" Resultados guardados en: results/cpu_local/eda/")
        print(" Gráficos generados:")
        print("   • class_distribution.png")
        print("   • quality_analysis.png")
        print("   • visual_patterns.png")
        print("   • imbalance_analysis.png")
        print("📋 Reporte completo: reporte_eda_completo.txt")
        
    except Exception as e:
        print(f" Error durante el análisis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_complete_eda()