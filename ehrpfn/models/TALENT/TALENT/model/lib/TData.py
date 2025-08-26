import torch
from torch.utils.data import Dataset

class TData(Dataset):
    def __init__(self, is_regression, X, Y, y_info, part):
        assert(part in ['train', 'val', 'test'])
        X_num, X_cat = X
        self.X_num = X_num[part] if X_num is not None else None
        self.X_cat = X_cat[part] if X_cat is not None else None
        if Y is not None:
            self.Y = Y[part]
        else:
            self.Y = None
        self.y_info = y_info
        self.num_items = len(self.Y) if self.Y is not None else len(self.X_num) if self.X_num is not None else len(self.X_cat) if self.X_cat is not None else 0
        # self.num_class = 1 if is_regression else torch.unique(Y['train']).shape[0]
        
    def get_dim_in(self):
        return 0 if self.X_num is None else self.X_num.shape[1]

    def get_categories(self):
        return (
            None
            if self.X_cat is None
            else [
                len(set(self.X_cat[:, i].cpu().tolist()))
                for i in range(self.X_cat.shape[1])
            ]
        )

    def __len__(self):
        return self.num_items
    
    def __getitem__(self, i):
        if self.X_num is not None and self.X_cat is not None:
            data = (self.X_num[i], self.X_cat[i])
        elif self.X_cat is not None and self.X_num is None:
            data = self.X_cat[i]
        else:
            data = self.X_num[i]
        if self.Y is not None:
            label = self.Y[i]
        else:
            label = -1 # should return empty
        return data, label

