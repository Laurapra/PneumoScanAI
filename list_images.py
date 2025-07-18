import os
from pathlib import Path

def list_available_images():
    
    print(" Explorando dataset disponible...")
    print("=" * 50)
    
    base_path = Path("data/chest_xray")
    
    if not base_path.exists():
        print(" Carpeta data/chest_xray no encontrada")
        return
    
    for split in ['train', 'val', 'test']:
        split_path = base_path / split
        if split_path.exists():
            print(f"\n {split.upper()}:")
            
            for class_name in ['NORMAL', 'PNEUMONIA']:
                class_path = split_path / class_name
                if class_path.exists():
                    images = list(class_path.glob('*.jpeg')) + list(class_path.glob('*.jpg'))
                    print(f"   {class_name}: {len(images)} imágenes")
                    
                    if images:
                        print(f"      Ejemplo: {images[0].name}")
                        if len(images) > 1:
                            print(f"      Ejemplo: {images[1].name}")
                    print()
        else:
            print(f"❌ {split} no encontrado")
    
    # Mostrar algunas rutas completas de ejemplo
    test_pneumonia = base_path / "test" / "PNEUMONIA"
    if test_pneumonia.exists():
        images = list(test_pneumonia.glob('*.jpeg'))[:5]
        print(" RUTAS DE EJEMPLO PARA USAR:")
        for img in images:
            print(f"   {img}")

if __name__ == "__main__":
    list_available_images()