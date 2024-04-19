import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch .utils.data import  DataLoader
from torchvision.transforms import ToTensor,Resize, Normalize
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import shutil
from skimage.transform import resize
from matplotlib import image as mpimg
from torch.utils.data import random_split



class Model_evaluation:
    def __init__(self, general_param, weights_init_func):
        self.general_param = general_param
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_init_func = weights_init_func
        
    def _init_model(self, train_param):
        self.model = train_param['model'](img_channel = self.general_param['img_channel'], num_classes =  self.general_param['num_classes']).to(self.device)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model.apply(self.weights_init_func)
        
    def _init_optimizer_scheduler(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.general_param['lr'])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size = self.general_param['scheduler_step'], gamma = self.general_param['scheduler_gamma'])
        
    def _init_criterion(self):
        self.criterion = nn.CrossEntropyLoss()
        
    def _train(self, dataloader):
        self.model.train()
        torch.autograd.set_detect_anomaly(True)
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
        
    def _evaluate_model(self, data_loader):
        correct, loss = 0, 0
        for data, target in data_loader:
              data, target = data.to(self.device), target.to(self.device)
              output = self.model(data)
              _, pred = torch.max(output,1)
              loss += self.criterion(output, target).item()
              correct += pred.eq(target.data.view_as(pred)).sum()
        return loss, correct

    def _test(self, train_dataloader, test_dataloader):
        self.model.eval()
        with torch.no_grad():
          loss, correct = self._evaluate_model(train_dataloader)
          loss /= len(train_dataloader.dataset)
          train_accuracy = correct/len(train_dataloader.dataset)
          print(f'train loss: {loss}, acc: {train_accuracy}')

          
          loss, correct = self._evaluate_model(test_dataloader)
          loss /= len(test_dataloader.dataset)
          print(f'test loss: {loss}, acc: {correct/len(test_dataloader.dataset)}')
          return ((train_accuracy).cpu().numpy(),(correct/len(test_dataloader.dataset)).cpu().numpy())
    
    def _plot_accuracy(self,train_history,train_param, model_save_path):
        train_scores, test_scores = zip(*train_history)

        epochs = range(1, len(train_scores) + 1)

        plt.plot(epochs, train_scores, label='Training accuracy')
        plt.plot(epochs, test_scores, label='Testing accuracy')
        plt.text(max(epochs), min(min(train_scores), min(test_scores)), 'Training mean (last 3):'+ str(round(np.mean(train_scores[-3:]),2))+ '\nTesting mean (last 3):'+ str(round(np.mean(test_scores[-3:]),2)),verticalalignment='bottom', horizontalalignment='right')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Model: '+ train_param['model_name'] +'    Batch size:  '+ str(train_param['batch_size']) + '    Number of words: ' + str(self.general_param['num_classes']))
        plt.legend()

        plt.savefig(model_save_path.replace('.pt','.png'))
        plt.show()

    def _save_model(self, train_param):
        if os.path.exists('models'):
            shutil.rmtree('models')
        os.mkdir('models')
        model_save_path = os.path.join('models', train_param['model_name'] +'_'+ str(train_param['batch_size']) +'_'+ str(self.general_param['num_classes'])+'.pt')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, model_save_path)
        return model_save_path
    
    def train_model(self, n_epochs, train_param, train_loader, test_loader):
        self._init_model(train_param)
        self._init_optimizer_scheduler()
        self._init_criterion()
        train_history = []
        for epoch in range(1, n_epochs + 1):
              self._train(train_loader)
              print(f'Epoch {epoch}/{n_epochs}')
              print('-' * 10)
              train_history.append(self._test(train_loader, test_loader))
              self.scheduler.step()
            
        model_save_path = self._save_model(train_param)
        self._plot_accuracy(train_history, train_param, model_save_path)
        
