import torch
import torch.nn as nn
import torch.optim as optim
from main_experiment_utils import ResNet18, safe_load_dict, accuracy
import os


class CMA(object):
    """
    A continual moving average for tracking loss updates.
    """

    def __init__(self):
        self.N = 0
        self.avg = 0.0

    def update(self, X):
        self.avg = (X + self.N * self.avg) / (self.N + 1)
        self.N = self.N + 1


def build_classifier(classifier_ckpt, imagenet_weights=False, num_classes=100, fe=False):
    classifier = ResNet18(pretrained=imagenet_weights, num_classes=num_classes)
    if classifier_ckpt is None:
        print("Will not resume any checkpoints!")
    else:
        resumed = torch.load(classifier_ckpt, map_location='cuda:0')
        print("Resuming from {}".format(classifier_ckpt))
        if 'fc.weight' in resumed:
            safe_load_dict(classifier, resumed)
        else:
            if 'state_dict' in resumed:
                state_dict_key = 'state_dict'
            else:
                state_dict_key = 'model_state'
            safe_load_dict(classifier, resumed[state_dict_key])
    if fe:
        print('Using model as feature extractor')
        for params in classifier.model.parameters():
            params.requires_grad = False
        for params in classifier.model.fc.parameters():
            params.requires_grad = True
    return classifier


class ResNet18_Offline(object):

    def __init__(self, num_classes, classifier_ckpt=None, imagenet_weights=False, feature_extractor=False,
                 batch_size=128, init_lr=0.1, milestones=[15, 30], num_epochs=40, weight_decay=1e-4, save_dir=None,
                 save_ckpt=False):
        classifier = build_classifier(classifier_ckpt, imagenet_weights, num_classes, feature_extractor)
        self.model = classifier
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.milestones = milestones
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.save_dir = save_dir
        self.save_ckpt = save_ckpt
        self.num_classes = num_classes

    def fit(self, train_loader, test_loader, ix, verbose=True):
        model = self.model.cuda()
        model.train()
        criterion = nn.CrossEntropyLoss()
        best_acc = 0.

        msg = '\rEpoch (%d/%d) -- Iter (%d/%d) -- Train Loss=%1.6f'
        iters = int(len(train_loader.dataset) / self.batch_size)

        optimizer = optim.SGD(model.parameters(), lr=self.init_lr, momentum=0.9, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=0.1)

        for e in range(self.num_epochs):
            model = model.train()
            total_loss = CMA()
            c = 0
            for X_train, y_train in train_loader:
                X = X_train.cuda()
                y = y_train.cuda()

                output = model(X)
                loss = criterion(output, y)

                optimizer.zero_grad()  # zero out grads before backward pass because they are accumulated
                loss.backward()
                optimizer.step()
                total_loss.update(loss.item())
                if verbose:
                    print(msg % (e, self.num_epochs, c, iters, total_loss.avg), end="")
                c += 1
            scheduler.step(e)

            # compute test accuracies
            _, probas, y_test_init = self.predict(test_loader)
            top1, top5 = accuracy(probas, y_test_init, topk=(1, 5))
            print('\nEpoch (%d/%d) -- Test Accuracy: top1=%0.2f%% -- top5=%0.2f%%' % (e, self.num_epochs, top1, top5))

            if self.save_ckpt:
                torch.save({'state_dict': model.state_dict()},
                           f=os.path.join(self.save_dir, 'offline_resnet18_epoch%d_ix%d.pth' % (e, ix)))

                # save out best checkpoint
                if top1 > best_acc:
                    best_acc = top1
                    torch.save({'state_dict': model.state_dict()},
                               f=os.path.join(self.save_dir, 'offline_resnet18_best_ix%d.pth' % ix))

    def predict(self, data_loader):
        with torch.no_grad():
            self.model.eval()
            self.model.cuda()

            probas = torch.zeros((len(data_loader.dataset), self.num_classes))
            all_lbls = torch.zeros((len(data_loader.dataset)))
            start_ix = 0
            for batch_ix, batch in enumerate(data_loader):
                batch_x, batch_lbls = batch[0], batch[1]
                batch_x = batch_x.cuda()
                batch_lbls = batch_lbls.cuda()
                logits = self.model(batch_x)
                end_ix = start_ix + len(batch_x)
                probas[start_ix:end_ix] = torch.softmax(logits, dim=1)
                all_lbls[start_ix:end_ix] = batch_lbls.squeeze()
                start_ix = end_ix

            preds = probas.max(1)[1]

        return preds, probas, all_lbls.long()
