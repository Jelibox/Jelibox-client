"""
Dynamic Class Manager for Annotation Tool
Handles loading, saving, adding, and deleting classes with color generation
"""
import os
import random
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict
import colorsys

from . import workspace_config

class ClassManager:
    def __init__(self, workspace_name: str):
        self.workspace_name = workspace_name
        self.config_dir = "configs"
        self.class_file = workspace_config.config_path(workspace_name)
        self.yolo_dataset_root = f"YOLOdataset/{workspace_name}"
        self.labels_folder = os.path.join(self.yolo_dataset_root, "labels")
        self.data_yaml_path = os.path.join(self.yolo_dataset_root, "data.yaml")
        self.voc_dataset = f"vocdataset/{workspace_name}"
        
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.labels_folder, exist_ok=True)
        os.makedirs(self.voc_dataset, exist_ok=True)
        
        self.classes: List[str] = []
        self.removed_targets: List[str] = []  # YOLO-World targets dropped by the last save
        self.colors: List[Tuple[int, int, int]] = []
        
        # Load or initialize classes
        self._load_classes()
        self._generate_colors()
    
    def _load_classes(self):
        """Load classes from config or rebuild the config from VOC XML files."""
        existing = workspace_config.get_classes(self.workspace_name)
        if existing is not None:
            self.classes = existing
            print(f"[ClassManager] Loaded {len(self.classes)} classes from {self.class_file}")
        else:
            self.classes = self._scan_classes_from_xml()
            self._save_classes()
            if self.classes:
                print(f"[ClassManager] Rebuilt {self.class_file} from VOC XML annotations")
            else:
                print(f"[ClassManager] Created empty class file: {self.class_file}")

    def _scan_classes_from_xml(self) -> List[str]:
        """Collect unique class names from all VOC XML files in the workspace."""
        if not os.path.isdir(self.voc_dataset):
            return []

        classes = []
        known_classes = set()
        xml_files = sorted(
            filename for filename in os.listdir(self.voc_dataset)
            if filename.lower().endswith('.xml')
        )

        for xml_file in xml_files:
            xml_path = os.path.join(self.voc_dataset, xml_file)
            try:
                root = ET.parse(xml_path).getroot()
            except (ET.ParseError, OSError) as exc:
                print(f"[ClassManager] Could not scan {xml_file}: {exc}")
                continue

            for name_elem in root.findall('.//object/name'):
                class_name = (name_elem.text or '').strip()
                if class_name and class_name not in known_classes:
                    known_classes.add(class_name)
                    classes.append(class_name)

        print(f"[ClassManager] Found {len(classes)} classes in {self.voc_dataset}")
        return classes
    
    def _save_classes(self):
        """Save classes to file"""
        self.removed_targets = workspace_config.set_classes(self.workspace_name, self.classes)
        print(f"[ClassManager] Saved {len(self.classes)} classes to {self.class_file}")
    
    # Calm, distinguishable class colors (RGB) - used for the first classes;
    # beyond these, colors are generated.
    CLASS_PALETTE = ["#E5566D", "#E0A030", "#3FB8A0", "#4A9FE0",
                     "#8B7BE8", "#E8804A", "#7BB661", "#D9709F"]

    def _generate_colors(self):
        """Class colors: the fixed calm palette first, generated colors after."""
        self.colors = []
        random.seed(42)

        n = len(self.classes)

        for i in range(min(n, len(self.CLASS_PALETTE))):
            h = self.CLASS_PALETTE[i].lstrip('#')
            r, g, b = (int(h[k:k + 2], 16) for k in (0, 2, 4))
            self.colors.append((b, g, r))   # BGR for OpenCV

        for i in range(len(self.CLASS_PALETTE), n):
            # Hue terdistribusi merata (golden angle)
            h = (i * 0.61803398875) % 1.0  

            # Saturation & Value dijaga supaya gak pucat
            s = random.uniform(0.75, 0.95)   # warna pekat
            v = random.uniform(0.45, 0.75)   # hindari terlalu terang

            r, g, b = colorsys.hsv_to_rgb(h, s, v)

            # Convert ke 0-255 (BGR buat OpenCV)
            color = (int(b * 255), int(g * 255), int(r * 255))

            self.colors.append(color)

        print(f"[ClassManager] Generated {len(self.colors)} colors")
    
    def add_class(self, class_name: str) -> Tuple[bool, str]:
        """
        Add new class
        Returns: (success, message)
        """

        # Validation: check spaces and symbols
        if not class_name.replace('_', '').replace('-', '').isalnum():
            return False, "Class name can only contain letters, numbers, underscore (_) and dash (-)"

        # Validation: check if empty
        if not class_name.strip():
            return False, "Class name cannot be empty"

        # Validation: check duplicate (case sensitive)
        if class_name in self.classes:
            return False, f"Class '{class_name}' already exists"

        # Validation: check length
        if len(class_name) > 16:
            return False, "Class name is too long (max 16 characters)"
        
        # Tambah class di index paling akhir
        self.classes.append(class_name)
        self._save_classes()
        self._generate_colors()  # Regenerate colors
        self._update_data_yaml()  # Update data.yaml
        
        return True, f"Class '{class_name}' successfully added"
    
    def delete_class(self, class_name: str) -> Tuple[bool, str]:
        """
        Delete class and update all label files (YOLO and XML)
        Returns: (success, message)
        """
        if class_name not in self.classes:
            return False, f"Class '{class_name}' not found"
        
        old_index = self.classes.index(class_name)
        
        # Buat mapping perubahan index
        # Class sebelum yang dihapus tetap sama, class setelahnya turun 1
        index_mapping = {}
        for i, cls in enumerate(self.classes):
            if i < old_index:
                index_mapping[i] = i  # Tetap sama
            elif i > old_index:
                index_mapping[i] = i - 1  # Turun 1
            # i == old_index akan dihapus, tidak perlu mapping
        
        # Hapus class dari list
        self.classes.remove(class_name)
        self._save_classes()
        self._generate_colors()
        
        # Update semua label files (YOLO .txt)
        yolo_updated = self._update_yolo_label_files(old_index, index_mapping)
        
        # Update semua XML files
        xml_updated = self._update_xml_files(class_name)
        
        # Update data.yaml
        self._update_data_yaml()
        
        message = f"Class '{class_name}' deleted. Updated {yolo_updated} YOLO labels and {xml_updated} XML files updated"
        if self.removed_targets:
            message += f". Removed {len(self.removed_targets)} YOLO-World target class(es): {', '.join(self.removed_targets)}"
        return True, message
    
    def _update_yolo_label_files(self, deleted_index: int, index_mapping: Dict[int, int]) -> int:
        """
        Update all YOLO label files after class deletion
        Returns: number of files updated
        """
        if not os.path.exists(self.labels_folder):
            return 0
        
        updated_count = 0
        label_files = [f for f in os.listdir(self.labels_folder) if f.endswith('.txt')]
        
        for label_file in label_files:
            label_path = os.path.join(self.labels_folder, label_file)
            
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                new_lines = []
                modified = False
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        old_class_idx = int(parts[0])
                        
                        # Skip jika class yang dihapus
                        if old_class_idx == deleted_index:
                            modified = True
                            continue
                        
                        # Update index jika perlu
                        if old_class_idx in index_mapping:
                            new_class_idx = index_mapping[old_class_idx]
                            parts[0] = str(new_class_idx)
                            modified = True
                        
                        new_lines.append(' '.join(parts) + '\n')
                
                # Tulis ulang jika ada perubahan
                if modified:
                    with open(label_path, 'w') as f:
                        f.writelines(new_lines)
                    updated_count += 1
            
            except Exception as e:
                print(f"[ClassManager] Error updating YOLO {label_file}: {e}")
        
        print(f"[ClassManager] Updated {updated_count} YOLO label files")
        return updated_count
    
    def _update_xml_files(self, deleted_class_name: str) -> int:
        """
        Update all XML annotation files by removing objects with deleted class
        Returns: number of files updated
        """
        if not os.path.exists(self.voc_dataset):
            return 0
        
        updated_count = 0
        xml_files = [f for f in os.listdir(self.voc_dataset) if f.endswith('.xml')]
        
        for xml_file in xml_files:
            xml_path = os.path.join(self.voc_dataset, xml_file)
            
            try:
                # Parse XML
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                # Find all <object> elements
                objects_to_remove = []
                for obj in root.findall('object'):
                    name_elem = obj.find('name')
                    if name_elem is not None and name_elem.text == deleted_class_name:
                        objects_to_remove.append(obj)
                
                # Remove objects with deleted class
                if objects_to_remove:
                    for obj in objects_to_remove:
                        root.remove(obj)
                    
                    # Save updated XML
                    tree.write(xml_path, encoding='utf-8', xml_declaration=True)
                    updated_count += 1
                    print(f"[ClassManager] Removed {len(objects_to_remove)} object(s) from {xml_file}")
            
            except Exception as e:
                print(f"[ClassManager] Error updating XML {xml_file}: {e}")
        
        print(f"[ClassManager] Updated {updated_count} XML files")
        return updated_count
    
    def _update_data_yaml(self):
        """Update data.yaml with current classes"""
        if not os.path.exists(self.data_yaml_path):
            # Buat data.yaml baru jika belum ada
            train_path = os.path.join(self.yolo_dataset_root, "train/images")
            val_path = os.path.join(self.yolo_dataset_root, "val/images")
        else:
            # Parse existing data.yaml untuk ambil train dan val path
            train_path = ""
            val_path = ""
            try:
                with open(self.data_yaml_path, 'r') as f:
                    for line in f:
                        if line.startswith('train:'):
                            train_path = line.split('train:')[1].strip()
                        elif line.startswith('val:'):
                            val_path = line.split('val:')[1].strip()
            except Exception as e:
                print(f"[ClassManager] Error reading data.yaml: {e}")
                train_path = os.path.join(self.yolo_dataset_root, "train/images")
                val_path = os.path.join(self.yolo_dataset_root, "val/images")
        
        # Jika path masih kosong, set default
        if not train_path:
            train_path = os.path.join(self.yolo_dataset_root, "train/images")
        if not val_path:
            val_path = os.path.join(self.yolo_dataset_root, "val/images")
        
        # Tulis data.yaml
        with open(self.data_yaml_path, 'w') as f:
            f.write(f"train: {train_path}\n")
            f.write(f"val: {val_path}\n")
            f.write(f"nc: {len(self.classes)}\n")
            f.write(f"names: {self.classes}\n")
        
        print(f"[ClassManager] Updated {self.data_yaml_path}")
    
    def get_classes(self) -> List[str]:
        """Get list of classes"""
        return self.classes.copy()
    
    def get_colors(self) -> List[Tuple[int, int, int]]:
        """Get list of colors"""
        return self.colors.copy()
    
    def get_class_index(self, class_name: str) -> int:
        """Get index of class, returns -1 if not found"""
        try:
            return self.classes.index(class_name)
        except ValueError:
            return -1
    
    def refresh(self):
        """Reload classes from file"""
        self._load_classes()
        self._generate_colors()