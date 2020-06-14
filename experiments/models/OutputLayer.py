import torch
from torch import nn
import torch.nn.functional as F
from torch import optim


class CMA(object):
    def __init__(self):
        self.N = 0
        self.avg = 0.0

    def update(self, X):
        self.avg = (X + self.N * self.avg) / (self.N + 1)
        self.N = self.N + 1


class Output(nn.Module):

    def __init__(self, input_shape, num_classes, weight_decay=1e-4, lr=0.1, batch_size=256, gpu_id=0, seed=111,
                 pred_batch_size=512, ckpt_file=None, resume_ckpt=False, grad_clip=None):
        super(Output, self).__init__()

        self.seed = seed
        self.num_classes = num_classes
        self.l2_scale = weight_decay
        self.lr = lr
        self.batch_size = batch_size
        self.gpu_id = gpu_id
        self.pred_batch_size = pred_batch_size
        self.grad_clip = grad_clip

        # set seed for initialization
        torch.manual_seed(self.seed)

        # make the mlp model
        self.model = nn.Sequential()
        layer = nn.Linear(input_shape, num_classes)
        self.model.add_module('fc', layer)

        if resume_ckpt:
            state = torch.load(ckpt_file)
            if 'state_dict' in state:
                state_dict_key = 'state_dict'
            elif 'model_dict' in state:
                state_dict_key = 'model_dict'
            old_model_state = self.state_dict()
            new_model_state = state[state_dict_key]
            for name, param in new_model_state.items():
                if name not in old_model_state:
                    # print('%s not found in old model.' % name)
                    continue
                if isinstance(param, nn.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                if old_model_state[name].shape != param.shape:
                    print('Shape mismatch...ignoring %s' % name)
                    continue
                else:
                    old_model_state[name].copy_(param)
                    print('Param %s successfully copied!' % name)

        self.optimizer = optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.l2_scale)

    def forward(self, X):
        return self.model(X)

    def fit(self, X_train, y_train):
        model = self.cuda(self.gpu_id)
        model.train()

        criterion = nn.CrossEntropyLoss()

        X = X_train.cuda()
        y = y_train.cuda()

        if len(X.shape) < 2:
            output = model(X).unsqueeze(0)
        else:
            output = model(X)
        loss = criterion(output, y)

        self.optimizer.zero_grad()  # zero out grads before backward pass because they are accumulated
        loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
        self.optimizer.step()

    def predict(self, X, return_probas=False):
        with torch.no_grad():
            self.eval()
            model = self.cuda(self.gpu_id)
            output = F.softmax(model(X).data.cpu(), dim=1)

            if not return_probas:
                preds = output.data.max(1)[1]
                return preds
            else:
                return output

    def ood_predict(self, X, auroc=True):
        return self.predict(X, return_probas=True)
