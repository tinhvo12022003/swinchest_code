from torch.utils.data import (Dataset, DataLoader)
from PIL import Image
from torchvision import transforms
import torch
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self, images, labels, transforms_pipeline, processor):
        self.images = images
        self.labels = labels
        self.transforms_pipeline = None
        if transforms_pipeline:
            self.transforms_pipeline = transforms.Compose([
                transforms.Resize((224, 224)), 
                transforms.ToTensor()
            ])
        self.processor = processor
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image = Image.open(image).convert("RGB")
        if self.transforms_pipeline:
            image = self.transforms_pipeline(image)
        pixel_values = self.processor(image, return_tensors='pt').pixel_values
        return {
            "pixel_values" : pixel_values.squeeze(0), 
            "label" : torch.tensor(label, dtype=torch.float32).squeeze()
        } 

def create_ratio_data(dataset, labels, train_ratio, val_ratio, test_ratio, batch_size, transforms_pipeline, processor):
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=test_ratio, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio/(val_ratio+train_ratio), random_state=42)
    
    train_dataset = CustomDataset(X_train, y_train, transforms_pipeline, processor)
    val_dataset = CustomDataset(X_val, y_val, transforms_pipeline, processor)
    test_dataset = CustomDataset(X_test, y_test, transforms_pipeline, processor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader