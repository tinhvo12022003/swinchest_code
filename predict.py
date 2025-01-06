import torch 
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class Predict():
    def __init__(self, model, dataset, device='cpu', model_path='model_weights.pth',mlp=None, processor=None):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.model_path = model_path
        self.load_model()
        self.model.to(self.device)
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.mlp = mlp
        self.processor = processor
    
    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path, weights_only=True))
        print(f"Model loaded from {self.model_path}")
        
    def test(self):
        self.model.eval()
        all_labels = []
        all_preds = []
        threshold = 0.5  # Threshold for binary classification

        with torch.no_grad():
            for batch in tqdm(self.dataset):
                images, labels = batch['pixel_values'].to(self.device), batch['label'].to(self.device)
                outputs = self.model(images)
                predicted = (outputs > threshold).float()
                
                all_labels.append(labels.cpu().numpy())
                all_preds.append(predicted.cpu().numpy())

        all_labels = np.concatenate(all_labels, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='micro')
        recall = recall_score(all_labels, all_preds, average='micro')
        f1 = f1_score(all_labels, all_preds, average='micro')
        auc = roc_auc_score(all_labels, all_preds, average='macro', multi_class='ovr')

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")


    # def binary_to_class_name(self, binary_array):
    #     return self.mlp.inverse_transform(binary_array)

    # def predict(self, image_path, show=False):
    #     self.model.eval()
    #     threshold = 0.5
    #     if os.path.exists(image_path):
    #         image = Image.open(image_path).convert("RGB")
    #         image = self.transforms(image)
    #         image = self.processor(image, return_tensors='pt').pixel_values
    #         with torch.no_grad():
    #             output = self.model(image)
    #             predicted = (output > threshold).float() 
    #             predicted = predicted.cpu().numpy()

    #         classes_names = self.binary_to_class_name(predicted)      

    #         if show==False:
    #             print("Predicted classes: ".format(", ".join(classes_names)))
    #         else:
    #             plt.figure(figsize=(8, 8))
    #             plt.imshow(image)
    #             plt.title("Predicted classes: {}".format(", ".join(classes_names)))
    #             plt.axis('off')
    #             plt.show()
    #     else: 
    #         raise ValueError("File not found!") 
        
    
