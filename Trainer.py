import os
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

class Trainer:
    def __init__(self, model, criterion, optimizer, train_dataloader, val_dataloader, num_epochs, device, patience=5, device_ids=None):
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.nn.DataParallel(model, device_ids=device_ids).to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_epochs = num_epochs
        self.best_val_loss = float('inf')
        self.patience = patience
        self.patience_counter = 0

    def write_train_log(self, epoch, loss, output_file='train_log.txt'):
        if not os.path.exists(output_file):
            with open(output_file, 'w') as f:
                f.write("Epoch,Loss\n")
        with open(output_file, 'a') as f:
            f.write(f"Epoch:{epoch},Loss:{loss}\n")

    def write_val_log(self, epoch, val_loss, val_acc, val_acc_part, precision, recall, f1, auc, output_file='val_log.txt'):
        if not os.path.exists(output_file):
            with open(output_file, 'w') as f:
                f.write("Epoch,Val_Loss,Val_Acc,Val_Acc_Part,Precision,Recall,F1,AUC\n")
        with open(output_file, 'a') as f:
            f.write(f"Epoch:{epoch},Val_Loss:{val_loss},Val_Acc:{val_acc},Val_Acc_Part:{val_acc_part},Precision:{precision},Recall:{recall},F1:{f1},AUC:{auc}\n")

    def save_checkpoint(self, state, filename='best_checkpoint.pth.tar'):
        torch.save(state, filename)

    def load_checkpoint(self, checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        return start_epoch

    def evaluate(self, dataloader):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        correct_part = 0
        total = 0
        all_labels = []
        all_preds = []
        threshold = 0.5

        with torch.no_grad():
            for batch in tqdm(dataloader):
                images = batch['pixel_values'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                predicted = (outputs > threshold).float()
                correct += (predicted == labels).all(dim=1).sum().item()
                correct_part += (predicted == labels).sum().item()
                total += labels.size(0)

                all_labels.append(labels.cpu().numpy())
                all_preds.append(predicted.cpu().numpy())

        val_loss /= len(dataloader)
        all_labels = np.concatenate(all_labels, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)

        precision = precision_score(all_labels, all_preds, average='micro')
        recall = recall_score(all_labels, all_preds, average='micro')
        f1 = f1_score(all_labels, all_preds, average='micro')
        auc = roc_auc_score(all_labels, all_preds, average='macro', multi_class='ovr')

        val_acc = correct / total
        val_acc_part = correct_part / (total * labels.size(1))

        return val_loss, val_acc, val_acc_part, precision, recall, f1, auc

    def resume_training(self, checkpoint_path):
        start_epoch = self.load_checkpoint(checkpoint_path)
        self.train(start_epoch=start_epoch)

    def train(self, start_epoch=0):
        for epoch in range(start_epoch, self.num_epochs):
            self.model.train()
            running_loss = 0.0
            for batch in tqdm(self.train_dataloader):
                images = batch['pixel_values'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            
            self.write_train_log(epoch, running_loss/len(self.train_dataloader))
            print(f"Epoch:{epoch+1},Training loss:{running_loss/len(self.train_dataloader)}")
            
            val_loss, val_acc, val_acc_part, precision, recall, f1, auc = self.evaluate(self.val_dataloader)
            self.write_val_log(epoch, val_loss, val_acc, val_acc_part, precision, recall, f1, auc)
            print(f"Epoch:{epoch+1},Validation loss:{val_loss}")

            # Save the best model based on validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_val_loss': self.best_val_loss
                }, 'best_checkpoint.pth.tar')
                print("Best model saved with validation loss: {:.4f}".format(val_loss))
                torch.save(self.model.state_dict(), 'model_weights.pth')
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print("Early stopping triggered")
                    break