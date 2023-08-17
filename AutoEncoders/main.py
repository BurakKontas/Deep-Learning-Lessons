import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the datasets
movies = pd.read_csv("./ml-1m/movies.dat", sep="::", header=None, engine='python', encoding="latin-1")# MovieID::Title::Genres
users = pd.read_csv("./ml-1m/users.dat", sep="::", header=None, engine='python', encoding="latin-1") # UserID::Gender::Age::Occupation::Zip-code
ratings = pd.read_csv("./ml-1m/ratings.dat", sep="::", header=None, engine='python', encoding="latin-1") # UserID::MovieID::Rating::Timestamp  

# Preparing the training set and the test set
training_set = pd.read_csv("./ml-100k/u1.base", sep="\t") # UserID    MovieID   Rating   Timestamp
training_set = np.array(training_set, dtype="int")
test_set = pd.read_csv("./ml-100k/u1.test", sep="\t") # UserID    MovieID   Rating   Timestamp  
test_set = np.array(test_set, dtype="int")

# Getting the number of users and movies
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data: pd.DataFrame):
    new_data = []
    for user_id in range(1, nb_users + 1):
        movie_ids = data[:, 1][data[:, 0] == user_id]
        ratings_ids = data[:, 2][data[:, 0] == user_id]
        ratings = np.zeros(nb_movies)
        ratings[movie_ids - 1] = ratings_ids
        new_data.append(ratings)
    
    return new_data        

training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the Neural Network
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(in_features=nb_movies, out_features=20) # out_features = first hidden neuron count
        self.fc2 = nn.Linear(in_features=20, out_features=10)
        self.fc3 = nn.Linear(in_features=10, out_features=20)
        self.fc4 = nn.Linear(in_features=20, out_features=nb_movies)
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
    
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay=0.5)

# Training the SAE
nb_epoch = 200

for epoch in range(1, nb_epoch + 1):
    train_loss = 0.
    counter = 0.
    for user_id in range(nb_users):
        input_var = Variable(training_set[user_id]).unsqueeze(0)
        target_var = input_var.clone()
        if torch.sum(target_var.data > 0) > 0:
            output = sae(input_var) 
            target_var.requires_grad = False
            output[target_var == 0] = 0
            loss = criterion(output, target_var)
            mean_corrector = nb_movies/float(torch.sum(target_var.data > 0) + 1e-10)
            loss.backward() # decides the direction
            train_loss += np.sqrt(loss.item() * mean_corrector)
            counter += 1.
            optimizer.step() # decides the amount
    print('epoch: '+str(epoch)+'loss: '+ str(train_loss/counter))
    
test_loss = 0
counter = 0.
for user_id in range(nb_users):
  input_var = Variable(training_set[user_id]).unsqueeze(0)
  target_var = Variable(test_set[user_id]).unsqueeze(0)
  if torch.sum(target_var.data > 0) > 0:
    output = sae(input_var)
    target_var.require_grad = False
    output[target_var == 0] = 0
    loss = criterion(output, target_var)
    mean_corrector = nb_movies/float(torch.sum(target_var.data > 0) + 1e-10)
    test_loss += np.sqrt(loss.item()*mean_corrector)
    counter += 1.
print('test loss: '+str(test_loss/counter))
