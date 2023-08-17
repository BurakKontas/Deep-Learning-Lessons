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


# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Creating the architecture of the Neural Network
class RBM():
    def __init__(self, nv, nh): # number of visible nodes, number of hidden nodes
        self.weight = torch.randn(nh, nv)
        self.hidden_bias = torch.randn(1, nh) # batch, bias
        self.visible_bias = torch.randn(1, nv) # batch, bias 
    
    def sample_hidden(self, x: torch.Tensor):
        wx = torch.mm(x, self.weight.t()) # t = transpose
        activation = wx + self.hidden_bias.expand_as(wx) # to make sure bias applied to every mini bias
        p_h_given_v = torch.sigmoid(activation) # probability
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    def sample_visible(self, y: torch.Tensor):
        wy = torch.mm(y, self.weight) # t = transpose
        activation = wy + self.visible_bias.expand_as(wy) # to make sure bias applied to every mini bias
        p_v_given_h = torch.sigmoid(activation) # probability
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    def train(self, v0: torch.Tensor, vk: torch.Tensor, ph0: torch.Tensor, phk: torch.Tensor):
        self.weight +=  (torch.mm(v0.t(),ph0) - torch.mm(vk.t(),phk)).t()
        self.visible_bias += torch.sum((v0 - vk), 0)
        self.hidden_bias += torch.sum((ph0 - phk), 0)
        
nv = len(training_set[0]) # number of visible nodes
nh = 100 # number of hidden nodes
batch_size = 100 # to update weights after each batch of users
rbm = RBM(nv, nh)

# Training the RBM
nb_epochs = 10
for epoch in range(1, nb_epochs + 1):
    train_loss = 0
    counter = 0.
    for user_id in range(0, nb_users - batch_size, batch_size):
        print(user_id)
        vk = training_set[user_id:user_id + batch_size]
        v0 = training_set[user_id:user_id + batch_size]
        ph0, _ = rbm.sample_hidden(v0)
        for k in range(10):
            # Markov Chain Monte Carlo Technique
            _, hk = rbm.sample_hidden(vk)
            _, vk = rbm.sample_visible(hk)
            vk[v0 < 0] = v0[v0 < 0]
        phk, _ = rbm.sample_hidden(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0])) # Changes technique to technique
        counter += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/counter))
    
test_loss = 0
counter = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_hidden(v)
        _,v = rbm.sample_visible(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        counter += 1.
print('test loss: '+str(test_loss/counter))