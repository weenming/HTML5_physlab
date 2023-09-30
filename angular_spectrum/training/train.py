import torch
from torch import nn
from data.load_minist import get_minist
from tqdm import tqdm


class RandomIndexer():
    def __init__(self, batch_size, n, seed=None):
        self.batch_size = batch_size
        self.n = n
        self.seed = seed

    def __iter__(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)
        self.idx = torch.randperm(self.n)
        self.i = 0
        return self

    def __next__(self):
        if self.i * self.batch_size > self.n - 1:
            raise StopIteration
        
        if (self.i + 1) * self.batch_size > self.n - 1:
            batch_idx = self.idx[self.i * self.batch_size: ]
            self.i += 1
            return batch_idx
        
        else:
            batch_idx = self.idx[self.i * self.batch_size:\
                                  (self.i + 1) * self.batch_size]
            self.i += 1
            return batch_idx

    __call__ = __next__



def train_iter(
        X_train, 
        y_train, 
        model, 
        loss_fn, 
        optimizer, 
        random_indexer, 
        pbar=None
):
    batch_num = y_train.size(0) // batch_idx.size(0)

    model.train()
    for i, batch_idx in enumerate(random_indexer):
        X, y = X_train[batch_idx, :, :], y_train[batch_idx]

        out = model(X)

        optimizer.zero_grad()
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        if pbar is not None and i % (batch_num // 10) == 0:
            pbar.set_description(f'loss: {loss}')


def eval_acc(X_test, y_test, model, batch_size):
    model.eval()
    correct = 0
    for i in range(y_test.size(0) // batch_size + 1):
        batch_idx = torch.arange(
            i * batch_size, 
            min(y_test.size(0), (i + 1) * batch_size)
        ).int()
        out = model(X_test[batch_idx, :, :])
        # print(out.argmax(-1))
        # print(y)
        correct += (out.argmax(-1) == y_test[batch_idx]).sum()
    return correct / y_test.size(0)
    

def fit(
        model: nn.Module, 
        X_train, 
        X_test, 
        y_train, 
        y_test, 
        epochs, 
        batch_size=64, 
        loss_fn=nn.CrossEntropyLoss(), 
        **optim_kwargs
):
    optimizer = torch.optim.Adam(model.parameters(), **optim_kwargs)
    
    for i in (pbar := tqdm(range(epochs))):
        
        indexer = RandomIndexer(batch_size, X_train.size(0), seed=i)
        train_iter(X_train, y_train, model, loss_fn, optimizer, indexer, pbar)

        print('acc: ', eval_acc(X_test, y_test, model, batch_size).item())
    
