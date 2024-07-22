#--------------------------Librerias--------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch import nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import pickle
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision.io import read_image
from torchvision.transforms import ToTensor, Lambda, Compose
import random
import os

#------------------------------------------------------------------------------
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
#transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 35
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

#--------------------------Datos para entrenar y testear--------------------------
train_set_orig = datasets.FashionMNIST('MNIST_data/', download = True, train = True,  transform = transform)
valid_set_orig = datasets.FashionMNIST('MNIST_data/', download = True, train = False, transform = transform)
#---------------------------------------------------------------------------------
train_ds, val_ds = random_split(train_set_orig, [50000, 10000])# Dividimos Conjunto entrada

#---------------------Modificamos Datasets para usar la MSE la misma toma vectores
class CustomDataser(Dataset):
    def __init__(self,dataset):
        self.dataset=dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self,i):
        image, label = self.dataset[i]
        input= image
        label = torch.flatten(image)
        return input,label
    
train_set =CustomDataser(train_ds)
valid_set =CustomDataser(val_ds)
test_set =CustomDataser(valid_set_orig)

#Creamos los dataloaders

batch_size= 1000
train_loader=DataLoader(train_set, batch_size=batch_size, shuffle= True)
valid_loader=DataLoader(valid_set, batch_size=batch_size, shuffle= True)
test_loader=DataLoader(test_set, batch_size=batch_size, shuffle= True)

#----------------------------------------------------------------------------------

p=0.1


class Autoencoder(nn.Module):
  def __init__(self):
    super(Autoencoder, self).__init__()
    self.flatten = nn.Flatten() 
    self.linear1 = nn.Linear(28*28, 64)
    self.linear4 = nn.Linear(64, 28*28)
    self.dropout1 = nn.Dropout(p=0.1)
  def forward(self, x_batch):
    outputs = x_batch.reshape(-1, 784)
    outputs= self.flatten(outputs) 
 #   outputs =self.dropout1(F.relu(self.linear4(self.dropout1(F.relu(self.linear1(outputs))))))
    outputs =self.dropout1(F.relu(self.linear4(self.dropout1(F.relu(self.linear1(outputs))))))
    
    return outputs



#creamos una instancia de una función perdida

loss_fn=nn.MSELoss() 
#loss_fn=nn.NLLLoss()
#creamos el modelo
model=Autoencoder()
#Creamos un optimizador, un stochastic Gradiente Descente o un Adam_

optimizer=optim.SGD(model.parameters(),lr=0.1,momentum=0.9)
#optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate,eps=1e-03,weight_decay=0,amsgrad= False)

#Entrenamiento
def train_loop(dataloader,model,loss_fn,optimizer):
#    model.train(True)
    num_batches =len(dataloader)
    sum_loss=0
    avg_loss=0
    pred=0
    true=0
    for batch_size, (X,y) in enumerate(dataloader):
    #calculamos la prediccion del modelo y la correspondiente perdida (error)
        optimizer.zero_grad()
        pred= model(X)
        true=y
        loss=loss_fn(pred,true)#---------------------
    #backpropagamos usando el optimizador proveido

        loss.backward()
        optimizer.step()
        sum_loss += loss.item()#*X.size(0)
    avg_loss = sum_loss / num_batches # loss per batch
    print(f'Test Error: Av loss: {avg_loss:>8f} \n')
    return avg_loss


#Validacion
def valid_loop(dataloader,model,loss_fn):
# model.eval()
     num_batches =len(dataloader)
     sum_los=0
     #loss_value=0
     for batch_size, (X,y) in enumerate(dataloader):
        #calculamos la prediccion del modelo y la correspondiente perdida (error)
       pred= model(X)
       loss_batch=loss_fn(pred,y).item()
       sum_los +=loss_batch
       #loss_value=loss_fn(pred,true)
      # sum_los+=(1/(batch_size+1))*(loss_value.data.item()-sum_los)
        #backpropagamos usando el optimizador proveido
     avg_los= sum_los/num_batches
     print(f'Test Error: Avg loss: {avg_los:>8f} \n')
     return avg_los



#Finalmente entrenamos iterando sobre epocas
#Ademas testeamoes el modelo en cada una de ellas
num_epochs=100
list_avg_train_loss_incorrecta = []
list_avg_train_loss=[]

list_avg_valid_loss=[]
for epoch in range(num_epochs):
    print(f'Época {epoch+1}\n............')
    model.train()
    avg_train_loss_incorrecta= train_loop(train_loader,model,loss_fn,optimizer)
    model.eval()
    with   torch.no_grad():
     avg_train_loss= valid_loop(train_loader,model,loss_fn) 
     avg_valid_loss=valid_loop(valid_loader,model,loss_fn)       
    list_avg_train_loss_incorrecta.append(avg_train_loss_incorrecta)
    list_avg_train_loss.append(avg_train_loss)
    list_avg_valid_loss.append(avg_valid_loss)    
print("Done!")

n = len(list_avg_train_loss_incorrecta)
figure = plt.figure()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(np.arange(n)+0.5,list_avg_train_loss_incorrecta,label='train-inco')
#plt.plot(list(range(1,len(list_avg_train_loss)+1)),list_avg_train_loss,label='valid')
plt.plot(list(range(1,len(list_avg_valid_loss)+1)),list_avg_valid_loss,label='tvalid')
plt.title('')
plt.legend()
plt.show()

with open('train64.pickle', 'wb') as f1:
    pickle.dump(list_avg_train_loss_incorrecta, f1)

with open('trainL64.pickle', 'wb') as f2:
    pickle.dump(list_avg_train_loss, f2)

with open('valid64.pickle', 'wb') as f3:
    pickle.dump(list_avg_valid_loss, f3)

# Probrando el modelo ya entrenado

model.eval()
with torch.no_grad():
    dataiter = iter(test_loader)
    images, _ = next(dataiter)

    original_images = images.view(-1,28,28)[:9]
    plt.figure(figsize=(20,4))
    for i in range(9):
        plt.subplot(2,10,i+1)
        plt.imshow(original_images[i].numpy(), cmap = 'gray')
        plt.title('Original')
        plt.axis('off')
    
    outputs = model(images.view(images.size(0),-1))
    reconstructed_images = outputs.view(-1,28,28)[:9]
    for i in range(9):
        plt.subplot(2,10,i+11)
        plt.imshow(reconstructed_images[i].numpy(), cmap = 'gray')
        plt.title('Reconstruida')
        plt.axis('off')
    plt.show()
