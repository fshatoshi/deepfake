#importation of essential libraries

import random
import torchvision.transforms as transforms
from PIL import Image
import os 
from torch.utils.data import Dataset , DataLoader


#Definition of the dataset class and data manager class

class FaceDataset(Dataset):
    
    
    def __init__(self, data_dir, is_train=True, items=None, class_to_idx=None, max_samples_per_class=None, max_classes=None, selected_persons=None):
        self.data_dir = data_dir
        self.images = []
        self.labels = []
        self.class_to_idx = {} if class_to_idx is None else class_to_idx
        
        # Transformations — train: augmentation for better generalization; val: deterministic
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], 
                                   [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], 
                                   [0.229, 0.224, 0.225])
            ])
        
        if items is not None:
            for path, label in items:
                self.images.append(path)
                self.labels.append(label)
        else:
            if selected_persons is not None:
                personnes = [p for p in selected_persons if os.path.isdir(os.path.join(data_dir, p))]
            else:
                personnes = sorted(
                    d for d in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir, d))
                )
            
            # Limit number of classes if specified
            if max_classes is not None and len(personnes) > max_classes:
                personnes = personnes[:max_classes]
            
            for idx, personne in enumerate(personnes):
                self.class_to_idx[personne] = idx
                dossier = os.path.join(data_dir, personne)
                
                # Collect images for this person
                person_images = []
                for img in os.listdir(dossier):
                    if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                        person_images.append(os.path.join(dossier, img))
                
                # Limit samples per class if specified
                if max_samples_per_class is not None and len(person_images) > max_samples_per_class:
                    random.shuffle(person_images)
                    person_images = person_images[:max_samples_per_class]
                
                for img_path in person_images:
                    self.images.append(img_path)
                    self.labels.append(idx)
        
        print(f" {len(self.images)} images, {len(self.class_to_idx)} persons")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        return self.transform(img), label


class DataManager:
    def __init__(self, data_dir, batch_size=32, val_ratio=0.2, split_seed=42, max_samples_per_class=None, max_classes=None, selected_persons=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.split_seed = split_seed
        self.max_samples_per_class = max_samples_per_class
        self.max_classes = max_classes
        self.selected_persons = selected_persons
    
    def get_dataloaders(self):
        full = FaceDataset(self.data_dir, is_train=True, max_samples_per_class=self.max_samples_per_class, max_classes=self.max_classes, selected_persons=self.selected_persons)
        n = len(full)
        if n == 0:
            raise ValueError(f"No images found under {self.data_dir}")
        
        items = list(zip(full.images, full.labels))
        rng = random.Random(self.split_seed)
        rng.shuffle(items)
        
        val_size = max(1, int(n * self.val_ratio))
        if val_size >= n:
            val_size = n - 1
        train_size = n - val_size
        if train_size < 1:
            train_items = items
            val_items = items
        else:
            train_items = items[:train_size]
            val_items = items[train_size:]
        
        train_dataset = FaceDataset(
            self.data_dir, is_train=True, items=train_items, class_to_idx=full.class_to_idx
        )
        val_dataset = FaceDataset(
            self.data_dir, is_train=False, items=val_items, class_to_idx=full.class_to_idx
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=False
        )
        
        return train_loader, val_loader, full.class_to_idx
