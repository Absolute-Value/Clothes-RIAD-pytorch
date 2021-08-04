import numpy as np
import torch


def denormalization(x):
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    # x = (x.transpose(1, 2, 0) * 255.).astype(np.uint8)
    return x


def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


class EarlyStop():
    """Used to early stop the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, verbose=True, delta=0, save_name="checkpoint.pt"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            save_name (string): The filename with which the model and the optimizer is saved when improved.
                            Default: "checkpoint.pt"
        """
        self.patience = patience
        self.verbose = verbose
        self.save_name = save_name
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, optimizer):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print((f'EarlyStopping counter: {self.counter} out of {self.patience}'))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, val_loss, model, optimizer):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print((f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'))
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(state, self.save_name)
        self.val_loss_min = val_loss

def gen_mask(k_list, n, im_size):
    while True:
        Ms = []
        for k in k_list:
            N = im_size // k # Ex. 256 / 4 = 64
            rdn = np.random.permutation(N**2) # 0 ~ 64x64-1 の配列を作って並び替え
            additive = N**2 % n # 64^2をn=3で割った時の余り
            if additive > 0: # 余りが有ったら
                rdn = np.concatenate((rdn, np.asarray([-1] * (n - additive)))) # n=3 - 余り分、-1を結合
            n_index = rdn.reshape(n, -1) # 配列をn=3等分
            for index in n_index:
                tmp = [0 if i in index else 1 for i in range(N**2)] # 0 ~ 64x64-1を回してn_indexにあったら0、なかったら1を入れる
                tmp = np.asarray(tmp).reshape(N, N) # 64x64にreshape
                tmp = tmp.repeat(k, 0).repeat(k, 1) # 配列の要素を繰り返して画像サイズと同じにする(64x64→256x256)
                Ms.append(tmp)
        yield Ms