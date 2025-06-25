# Modelos CNN para Detección de Neumonía
# Implementación de baseline y transfer learning

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50, DenseNet121, InceptionV3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np

class PneumoniaModels:  
    """Clase para manejar diferentes modelos de detección de neumonía"""
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.histories = {}  
    
    def create_baseline_cnn(self): 
        """Modelo CNN baseline simple"""
        print(" Creando modelo CNN baseline...")
        model = keras.Sequential([
            # Primera capa convolucional
            layers.Conv2D(32, (3, 3), activation='relu',
                         input_shape=(*self.config.IMG_SIZE, 3)),
            layers.MaxPooling2D(2, 2),
            layers.BatchNormalization(),
            
            # Segunda capa convolucional
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.BatchNormalization(),
            
            # Tercera capa convolucional
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.BatchNormalization(),
            
            # Cuarta capa convolucional
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.BatchNormalization(),
            
            # Aplanar y capas densas
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')  # Clasificación binaria
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.config.LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.models['baseline'] = model
        print(f"✓ Modelo baseline creado: {model.count_params():,} parámetros")
        return model
    
    def create_resnet_model(self, trainable_layers=0):
        """Modelo basado en ResNet50 con transfer learning"""
        print(" Creando modelo ResNet50...")
        
        # Cargar modelo pre-entrenado
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.config.IMG_SIZE, 3)
        )
        
        # Congelar capas base
        base_model.trainable = trainable_layers > 0
        if trainable_layers > 0:
            # Descongelar las últimas capas
            for layer in base_model.layers[:-trainable_layers]:
                layer.trainable = False
        
        # Crear modelo completo
        model = keras.Sequential([  
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
        
        # Usar learning rate más bajo para transfer learning
        lr = self.config.LEARNING_RATE * 0.1 if trainable_layers == 0 else self.config.LEARNING_RATE * 0.01
        
        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        model_name = f'resnet50_tl{trainable_layers}'
        self.models[model_name] = model
        print(f"✓ Modelo ResNet50 creado: {model.count_params():,} parámetros")
        print(f"  - Capas entrenables: {sum(1 for layer in model.layers if layer.trainable)}")
        return model
    
    def create_densenet_model(self, trainable_layers=0):
        """Modelo basado en DenseNet121 con transfer learning"""
        print(" Creando modelo DenseNet121...")
        
        # Cargar modelo pre-entrenado
        base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.config.IMG_SIZE, 3)
        )
        
        # Congelar capas base
        base_model.trainable = trainable_layers > 0
        if trainable_layers > 0:
            for layer in base_model.layers[:-trainable_layers]:
                layer.trainable = False
        
        # Crear modelo completo
        model = keras.Sequential([
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
        
        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        model_name = f'densenet121_tl{trainable_layers}'
        self.models[model_name] = model
        print(f'✓ Modelo DenseNet121 creado: {model.count_params():,} parámetros')
        return model
    
    def create_inception_model(self, trainable_layers=0):
        """Modelo basado en InceptionV3 con transfer learning"""
        print(" Creando modelo InceptionV3...")
        
        # Cargar modelo pre-entrenado
        base_model = InceptionV3(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.config.IMG_SIZE, 3)
        )
        
        # Congelar capas base
        base_model.trainable = trainable_layers > 0
        if trainable_layers > 0:
            for layer in base_model.layers[:-trainable_layers]:
                layer.trainable = False
        
        # Crear modelo completo
        model = keras.Sequential([
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
        
        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        model_name = f'inceptionv3_tl{trainable_layers}'
        self.models[model_name] = model
        print(f'✓ Modelo InceptionV3 creado: {model.count_params():,} parámetros')
        return model
    
    def get_callbacks(self, model_name):
        """Crear callbacks para el entrenamiento"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,  
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                f'models/{model_name}_best.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]
        return callbacks
    
    def train_model(self, model_name, train_generator, val_generator, class_weights=None, epochs=None):
        """Entrenar un modelo específico"""
        if model_name not in self.models:
            print(f" Modelo '{model_name}' no encontrado")
            return None
        
        model = self.models[model_name]
        epochs = epochs or self.config.EPOCHS
        
        print(f"\n Entrenando modelo: {model_name}")
        print(f"Épocas: {epochs}")
        print(f"Batch size: {self.config.BATCH_SIZE}")
        
        # Obtener callbacks
        callbacks = self.get_callbacks(model_name)
        
        # Entrenar modelo
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1  
        )
        
        self.histories[model_name] = history
        print(f" Entrenamiento de {model_name} completado")
        
        return history
    
    def plot_training_history(self, model_name):
        """Visualizar historial de entrenamiento"""
        if model_name not in self.histories:
            print(f" No hay historial para el modelo '{model_name}'")
            return
        
        history = self.histories[model_name].history
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Historial de Entrenamiento - {model_name}', fontsize=16)  # ❌ Era "subtitle"
        
        # Loss
        axes[0, 0].plot(history['loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(history['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Época')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(history['accuracy'], label='Train Acc', color='blue')
        axes[0, 1].plot(history['val_accuracy'], label='Val Acc', color='red')
        axes[0, 1].set_title('Accuracy')  
        axes[0, 1].set_xlabel('Época')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        if 'precision' in history:
            axes[1, 0].plot(history['precision'], label='Train Precision', color='blue')
            axes[1, 0].plot(history['val_precision'], label='Val Precision', color='red')
            axes[1, 0].set_title('Precision')
            axes[1, 0].set_xlabel('Época')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Recall
        if 'recall' in history:
            axes[1, 1].plot(history['recall'], label='Train Recall', color='blue')
            axes[1, 1].plot(history['val_recall'], label='Val Recall', color='red')
            axes[1, 1].set_title('Recall')
            axes[1, 1].set_xlabel('Época')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'plots/{model_name}_training_history.png', dpi=300, bbox_inches='tight')  
        plt.show()
    
    def load_best_model(self, model_name):
        """Cargar el mejor modelo guardado"""
        model_path = f'models/{model_name}_best.h5'
        try:
            model = keras.models.load_model(model_path)
            self.models[f'{model_name}_best'] = model
            print(f' Modelo {model_name} cargado desde {model_path}')
            return model
        except Exception as e:
            print(f' Error cargando modelo {model_name}: {e}')
            return None
    
    def get_model_summary(self, model_name):
        """Obtener resumen del modelo"""
        if model_name in self.models:
            print(f"\n Resumen del modelo: {model_name}")
            self.models[model_name].summary()
        else:
            print(f" Modelo '{model_name}' no encontrado")


def train_all_models(config, train_gen, val_gen, class_weights):
    """Entrenar todos los modelos del proyecto"""
    print(" INICIANDO ENTRENAMIENTO DE TODOS LOS MODELOS")
    print("=" * 50)
    
    # Inicializar gestor de modelos
    model_manager = PneumoniaModels(config)  
    
    # Lista de modelos a entrenar
    models_to_train = [  
        ('baseline', 'create_baseline_cnn', {}),
        ('resnet50_frozen', 'create_resnet_model', {'trainable_layers': 0}),
        ('resnet50_ft', 'create_resnet_model', {'trainable_layers': 20}),
        ('densenet121_frozen', 'create_densenet_model', {'trainable_layers': 0}),
        ('inceptionv3_frozen', 'create_inception_model', {'trainable_layers': 0}),
    ]
    
    results = {}
    
    for model_name, create_method, kwargs in models_to_train:
        print(f"\n{'='*60}")
        print(f" CONFIGURANDO MODELO: {model_name.upper()}")
        print(f"{'='*60}")
        
        # Crear modelo
        create_func = getattr(model_manager, create_method)
        model = create_func(**kwargs)
        
        # Entrenar modelo
        history = model_manager.train_model(
            model_name, 
            train_gen, 
            val_gen, 
            class_weights=class_weights,
            epochs=30 if 'baseline' in model_name else 20  # Menos épocas para transfer learning
        )
        
        if history:
            # Guardar métricas finales
            final_metrics = {
                'train_loss': history.history['loss'][-1],
                'val_loss': history.history['val_loss'][-1],
                'train_acc': history.history['accuracy'][-1],
                'val_acc': history.history['val_accuracy'][-1],
            }
            results[model_name] = final_metrics
            
            # Mostrar gráficos
            model_manager.plot_training_history(model_name)
    
    # Resumen de resultados
    print(f"\n{'='*60}")
    print(" RESUMEN DE RESULTADOS DE ENTRENAMIENTO")
    print(f"{'='*60}")
    
    import pandas as pd
    df_results = pd.DataFrame(results).T
    print(df_results.round(4))
    
    return model_manager, results


# Ejemplo de uso
if __name__ == "__main__":
    # Importar configuración (asumiendo que ya se ejecutó el setup inicial)
    from pneumonia_setup import Config, main as setup_main
    
    config = Config()
    
    # Ejecutar setup si es necesario
    train_gen, val_gen, test_gen, class_weights = setup_main()
    
    # Entrenar todos los modelos
    model_manager, results = train_all_models(config, train_gen, val_gen, class_weights)