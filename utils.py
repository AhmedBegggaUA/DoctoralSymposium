import torch
from sklearn.model_selection import train_test_split
from torch_geometric.utils import to_dense_adj
import numpy as np
def train(data,model,optimizer,criterion):
      model.train() # Set model to training mode.
      optimizer.zero_grad()  # Clear gradients.
      out,loss_norm = model(data)  # Perform a single forward pass.
      #Get the accuracy of the model
      pred = out.argmax(dim=1).squeeze(0)  # Use the class with highest probability.
      train_correct = pred[data.train_mask] == data.y[data.train_mask]  # Check against ground-truth labels.
      train_acc = int(train_correct.sum()) / int(data.train_mask.sum())  # Derive ratio of correct predictions.
      loss =  loss_norm + criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      optimizer.zero_grad()  # Clear gradients.
      return loss,train_acc
  
'''
      test function:
            * Inputs:
                  - adj: torch tensor of adjacency matrix
                  - data: torch tensor of data
                  - model: torch model
                  - test_mask: torch tensor of boolean values, True if data is in validation set 
            * Outputs:
                  - test_acc: float of validation accuracy
            * Description:
                  - performs the validation step of the model, only tacking into account the testing nodes.
'''    
@torch.no_grad()
def test(data,model):
      model.eval() # Set model to evaluation mode.
      out,_ = model(data) # Perform a single forward pass.
      pred = out.argmax(dim=1).squeeze(0)  # Use the class with highest probability.
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
      return test_acc
