import json
import os
import random

import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import Subset, DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, roc_auc_score

from StreamDataset import StreamDataset


class GaussianDataset(Dataset):
    def __init__(self, transform=None, class_label=0, length=1000, num_channels=3, width=224):
        self.name = 'Gaussian'
        self.transform = transform
        self.class_label = class_label
        self.length = length
        self.num_channels = num_channels
        self.width = width

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = torch.randn(self.num_channels, self.width, self.width)
        if self.transform:
            sample = self.transform(sample)
        return sample, torch.tensor(self.class_label)


def filter_by_class(labels, seen_classes):
    ixs = []
    for c in seen_classes:
        i = list(np.where(labels == c)[0])
        ixs += i
    return ixs


def get_stream_data_loader(images_dir, training, ordering=None, 
                           batch_size=128, shuffle=False, augment=False, 
                           num_workers=8, seen_classes=None, seed=10, ix=None):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if training and augment:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize])

    dataset = StreamDataset(images_dir, train=training, ordering=ordering, 
                            transform=transform, bbox_crop=True, ratio=1.10, 
                            seed=seed)
    labels = np.array([t for t in dataset.targets])

    if seen_classes is not None:
        indices = filter_by_class(labels, seen_classes)
        sub = Subset(dataset, indices)
        loader = DataLoader(sub, batch_size=batch_size, 
                            num_workers=num_workers, pin_memory=True, 
                            shuffle=shuffle)
    elif ix is not None:
        sub = Subset(dataset, ix)
        loader = DataLoader(sub, batch_size=batch_size, 
                            num_workers=num_workers, pin_memory=True, 
                            shuffle=shuffle)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, 
                            num_workers=num_workers, pin_memory=True, 
                            shuffle=shuffle)
    return loader


def accuracy(output, target, topk=(1,), output_has_class_ids=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    if not output_has_class_ids:
        output = torch.Tensor(output)
    else:
        output = torch.LongTensor(output)
    target = torch.LongTensor(target)
    with torch.no_grad():
        maxk = max(topk)
        batch_size = output.shape[0]
        if not output_has_class_ids:
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
        else:
            pred = output[:, :maxk].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def auosc_score(in_scores, out_scores, in_correct_label, steps=10000):
    """
    Compute the AUOSC metric that accounts for correct classification and OOD detection.
    :param in_scores: scores for in-distribution samples
    :param out_scores: scores for out-of-distribution samples
    :param in_correct_label: binary vector that says whether the model was correct for each in-distribution sample
    :param steps: number of steps to use for trapezoidal rule calculation
    :return: AUOSC, normalized AUOSC, correct classification rate, false positive rate
    """
    tmin = torch.min(torch.cat((in_scores, out_scores)))
    tmax = torch.max(torch.cat((in_scores, out_scores)))

    if tmin == tmax:
        print('\nWarning: all output scores for in and out samples are identical.')
    thresh_range = torch.arange(tmin, tmax, (tmax - tmin) / steps)
    FPR = []
    CCR = []
    for t in thresh_range:
        FPR.append(float(torch.sum(out_scores > t)) / out_scores.shape[0])
        CCR.append(float(torch.sum(in_correct_label[in_scores > t])) / in_scores.shape[0])
    ccr = np.array(CCR[::-1])
    fpr = np.array(FPR[::-1])
    return np.trapz(ccr, fpr), np.trapz(ccr / ccr.max(), fpr), ccr, fpr


def compute_k_plus_one_accuracy(true_labels, in_scores, out_scores, k_plus_one_tpr, num_classes):
    s_in, p_in = in_scores.max(dim=1)
    s_out, p_out = out_scores.max(dim=1)

    # num_samples = min(ood_num_samples, len(in_scores))
    num_samples = len(in_scores)

    thresh = np.sort(s_in.numpy())[int(num_samples * (1 - k_plus_one_tpr))]
    print('Threshold K+1: %0.2f' % thresh)

    in_labels = torch.where(s_in >= thresh, p_in.long(), torch.ones_like(s_in).long() * num_classes)
    out_labels = torch.where(s_out >= thresh, p_out.long(), torch.ones_like(s_out).long() * num_classes)
    preds = torch.cat((in_labels, out_labels))
    acc = accuracy_score(true_labels, preds.numpy())
    return acc

def save_accuracies(accuracies, index, save_path):
    name = 'accuracies_index_' + str(index) + '.json'
    json.dump(accuracies, open(os.path.join(save_path, name), 'w'))
    
def save_array(y_pred, index, save_path, save_name):
    name = save_name + '_index_' + str(index)
    torch.save(y_pred, save_path + '/' + name + '.pt')
    
def save_predictions(y_pred, index, save_path):
    name = 'preds_index_' + str(index)
    torch.save(y_pred, save_path + '/' + name + '.pt')

def select_indices(targets, classes, num_samples, seed=200):
    indices = []
    T = np.array(targets)
    samples_per_class = np.ones(len(classes), dtype=int) * int(num_samples / len(classes))
    samples_per_class[:num_samples - sum(samples_per_class)] += 1
    for n, c in enumerate(classes):
        ind = list(np.where(T == c)[0])
        random.seed(seed)
        random.shuffle(ind)
        indices.extend(ind[:samples_per_class[n]])
    if (len(indices)<num_samples) and (len(ind)>samples_per_class[-1]):
        extra_samples = num_samples-len(indices)
        indices.extend(ind[samples_per_class[-1]:samples_per_class[1]+extra_samples])
    return indices


def select_all_indices(targets, classes):
    indices = []
    T = np.array(targets)
    for n, c in enumerate(classes):
        ind = list(np.where(T == c)[0])
        indices.extend(ind)
    return indices


def get_ood_loaders(test_loader, num_classes, batch_size, ood_num_samples, seen_classes=None, dataset=None,
                    only_explicit_out_data=False):
    test_set = test_loader.dataset
    if ood_num_samples == -1:
        print('Using all test data for OOD')
    else:
        print('Num OOD samples ', ood_num_samples)

    if seen_classes is None:
        in_classes = np.arange(num_classes)

        if dataset == 'stream51':
            # since stream-51 has OOD samples not in its included classes
            out_classes = np.array([num_classes])
        else:
            out_classes = np.array(())
    else:
        print('Making in-loader from these classes: ', seen_classes)
        in_classes = np.array(seen_classes)

        if only_explicit_out_data:
            out_classes = np.array([num_classes])
        else:
            b = list(np.arange(num_classes))
            out_classes = np.array([item for item in b if item not in seen_classes])
            if dataset == 'stream51':
                # since stream-51 has OOD samples not in its included classes
                out_classes = np.append(out_classes, num_classes)

    if ood_num_samples == -1:
        in_indices = select_all_indices(test_set.targets, in_classes)
    else:
        in_indices = select_indices(test_set.targets, in_classes, ood_num_samples)
    in_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(test_set, in_indices), batch_size=batch_size,
                                            shuffle=False, num_workers=8, pin_memory=True)
    res_in_classes = list(np.unique(np.array(test_set.targets)[in_indices]))
    print('Inloader made from {} classes ({} samples)'.format(len(res_in_classes),len(in_indices)))
    print('Inloader made from {} classes ({} samples)'.format(len(res_in_classes),len(in_loader.dataset)))
    # out_loader = torch.utils.data.DataLoader(GaussianDataset(
    #     transforms.Normalize(mean=-1 * np.array([0.485, 0.456, 0.406]) / np.array([0.229, 0.224, 0.225]),
    #                          std=1 / np.array([0.229, 0.224, 0.225])), num_classes, ood_num_samples),
    #     batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    # out_loader_noise = torch.utils.data.DataLoader(
    #     GaussianDataset(transforms.Normalize(mean=np.array([0., 0., 0.]), std=np.array([1., 1., 1.])),
    #                     num_classes,
    #                     ood_num_samples), batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    if len(out_classes) != 0:

        if ood_num_samples == -1:
            out_indices = select_all_indices(test_set.targets, out_classes)
        else:
            out_indices = select_indices(test_set.targets, out_classes, ood_num_samples)
        out_loader_intra = torch.utils.data.DataLoader(torch.utils.data.Subset(test_set, out_indices),
                                                       batch_size=batch_size,
                                                       shuffle=False, num_workers=8, pin_memory=True)
        res_out_classes = list(np.unique(np.array(test_set.targets)[out_indices]))
        print('Outloader made from {} classes ({} samples)'.format(len(res_out_classes),len(out_indices)))

    else:
        out_loader_intra = None

    out_loader_noise = None

    return in_loader, out_loader_noise, out_loader_intra


def get_class_change_boundaries(labels):
    class_change_list = []
    prev_class = labels[0]
    for i, curr_class in enumerate(labels):
        if curr_class != prev_class:
            class_change_list.append(i)
            prev_class = curr_class
    return np.array(class_change_list)


def inspect_images(tr, test_loader_full):
    import matplotlib.pyplot as plt
    ix = 50000
    a = tr.dataset[ix][0]
    label = tr.dataset[ix][1]
    print('label ', label)
    b = np.moveaxis(a.numpy(), 0, -1)
    plt.imshow(b)
    plt.title('Train Label: ' + str(label))
    plt.show()
    ix = label * 50  # 50 images per class in test set
    a = test_loader_full.dataset[ix][0]
    label = test_loader_full.dataset[ix][1]
    print('label ', label)
    b = np.moveaxis(a.numpy(), 0, -1)
    plt.imshow(b)
    plt.title('Test Label: ' + str(label))
    plt.show()


def ood_detection(ood_scores_in, true_labels_in, ood_scores_out, true_labels_out, save_dir, index, num_classes,
                  k_plus_one_tpr, ood_type):
    # compute auosc scores
    s_in, p_in = ood_scores_in.max(dim=1)
    s_out, _ = ood_scores_out.max(dim=1)
    in_correct_label = p_in == true_labels_in
    auosc, auosc_norm, ccr, fpr = auosc_score(s_in, s_out, in_correct_label)

    num_in_samples = len(ood_scores_in.numpy())
    num_out_samples = len(ood_scores_out.numpy())

    # concatenate everything together
    y_true_ood = np.concatenate((np.ones(num_in_samples), np.zeros(num_out_samples)))
    y_true_actual_labels = np.concatenate((true_labels_in.numpy(), true_labels_out.numpy()))
    y_scores = np.concatenate((ood_scores_in.numpy(), ood_scores_out.numpy()))

    # save out data to calculate metrics offline
    save_array(y_true_actual_labels, index=index, save_path=save_dir,
                               save_name='ood_y_true_labels_' + ood_type)
    save_array(y_true_ood, index=index, save_path=save_dir, save_name='ood_y_true_binary_' + ood_type)
    save_array(y_scores, index=index, save_path=save_dir, save_name='ood_y_scores_' + ood_type)

    # compute K+1 accuracy score
    k_plus_one_acc = compute_k_plus_one_accuracy(y_true_actual_labels, ood_scores_in, ood_scores_out, k_plus_one_tpr,
                                                 num_classes)

    # compute auroc score and display all ood scores to user
    top_scores = np.max(y_scores, axis=1)
    auroc = roc_auc_score(y_true_ood, top_scores)
    print('\nOOD: AUROC=%0.2f -- AUOSC=%0.2f -- AUOSC Norm=%0.2f -- K+1 Acc=%0.2f' % (
        auroc, auosc, auosc_norm, k_plus_one_acc))
    return auroc, auosc, auosc_norm, k_plus_one_acc


def update_and_save_accuracies_ood(probas, y_test_init, ood_scores_in_intra, true_labels_in_intra, ood_scores_out_intra,
                                   true_labels_out_intra, save_dir, index, accuracies, num_classes, k_plus_one_tpr=0.5):
    top1, top5 = accuracy(probas, y_test_init, topk=(1, 5))
    print('\nIndex: %d -- top1=%0.2f%% -- top5=%0.2f%%' % (index, top1, top5))
    accuracies['top1'].append(float(top1))
    accuracies['top5'].append(float(top5))

    auroc_intra, auosc_intra, auosc_norm_intra, k_plus_one_acc_intra = ood_detection(ood_scores_in_intra,
                                                                                     true_labels_in_intra,
                                                                                     ood_scores_out_intra,
                                                                                     true_labels_out_intra,
                                                                                     save_dir, index, num_classes,
                                                                                     k_plus_one_tpr, 'intra')

    accuracies['auroc_score_intra'].append(float(auroc_intra))
    accuracies['auosc_score_intra'].append(float(auosc_intra))
    accuracies['auosc_norm_score_intra'].append(float(auosc_norm_intra))
    accuracies['k_plus_one_acc_intra'].append(float(k_plus_one_acc_intra))

    # save out results
    save_accuracies(accuracies, index=index, save_path=save_dir)
    save_predictions(probas, index=index, save_path=save_dir)


def update_and_save_accuracies(probas, y_test_init, save_dir, index, accuracies):
    top1, top5 = accuracy(probas, y_test_init, topk=(1, 5))
    print('\nIndex: %d -- top1=%0.2f%% -- top5=%0.2f%%' % (index, top1, top5))
    accuracies['top1'].append(float(top1))
    accuracies['top5'].append(float(top5))

    # save out results
    save_accuracies(accuracies, index=index, save_path=save_dir)
    save_predictions(probas, index=index, save_path=save_dir)

def safe_load_dict(model, new_model_state):
    old_model_state = model.state_dict()
    c = 0
    for name, param in new_model_state.items():
        n = name.split('.')
        beg = n[0]
        end = n[1:]
        if beg == 'module':
            name = '.'.join(end)
            name = 'model.' + name
        if beg[0] != 'm':
            name = 'model.' + '.'.join(n)
        if name not in old_model_state:
            # print('%s not found in old model.' % name)
            continue
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        c += 1
        if old_model_state[name].shape != param.shape:
            print('Shape mismatch...ignoring %s' % name)
            continue
        else:
            old_model_state[name].copy_(param)
    if c == 0:
        raise AssertionError('No previous ckpt names matched and the ckpt was not loaded properly.')

class ResNet18(nn.Module):
    def __init__(self, num_classes=100, pretrained=False, ckpt=None, lr=0.01, l2_scale=1e-4):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        if num_classes is not None:
            print('Changing output layer to contain %d classes.' % num_classes)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        if ckpt is None:
            print("Will not resume any checkpoints!")
        else:
            # resumed = torch.load(classifier_ckpt)
            resumed = torch.load(ckpt, map_location='cuda:0')
            if 'state_dict' in resumed:
                state_dict_key = 'state_dict'
            else:
                state_dict_key = 'model_state'
            print("Resuming from {}".format(ckpt))
            safe_load_dict(self, resumed[state_dict_key])

        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=l2_scale)

    def forward(self, x):
        return self.model(x)

    def fit(self, X_train, y_train):
        model = self.model.cuda()
        model.train()

        criterion = nn.CrossEntropyLoss()

        X = X_train.cuda()
        y = y_train.cuda()

        if len(X.shape) < 4:
            output = model(X.unsqueeze(0))
        else:
            output = model(X)
        loss = criterion(output, y)

        self.optimizer.zero_grad()  # zero out grads before backward pass because they are accumulated
        loss.backward()
        self.optimizer.step()

    def predict(self, X, return_probas=False):
        with torch.no_grad():
            self.model.eval()
            model = self.model.cuda()
            output = torch.softmax(model(X).data.cpu(), dim=1)

            if not return_probas:
                preds = output.max(1)[1]
                return preds
            else:
                return output

    def ood_predict(self, X, auroc=True):
        return self.predict(X, return_probas=True)


def get_name_to_module(model):
    name_to_module = {}
    for m in model.named_modules():
        name_to_module[m[0]] = m[1]
    return name_to_module


def get_activation(all_outputs, name):
    def hook(model, input, output):
        all_outputs[name] = output.detach()

    return hook


def add_hooks(model, outputs, output_layer_names):
    """

    :param model:
    :param outputs: Outputs from layers specified in `output_layer_names` will be stored in `output` variable
    :param output_layer_names:
    :return:
    """
    name_to_module = get_name_to_module(model)
    for output_layer_name in output_layer_names:
        name_to_module[output_layer_name].register_forward_hook(get_activation(outputs, output_layer_name))


class ModelWrapper(nn.Module):
    def __init__(self, model, output_layer_names, return_single=False):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.output_layer_names = output_layer_names
        self.outputs = {}
        self.return_single = return_single
        add_hooks(self.model, self.outputs, self.output_layer_names)

    def forward(self, x):
        self.model(x)
        output_vals = [self.outputs[output_layer_name] for output_layer_name in self.output_layer_names]
        if self.return_single:
            return output_vals[0]
        else:
            return output_vals
        
def build_classifier(classifier_ckpt, imagenet_weights=False, num_classes=100, fe=False):
    classifier = ResNet18(pretrained=imagenet_weights, num_classes=num_classes)
    if classifier_ckpt is None:
        print("Will not resume any checkpoints!")
    else:
        # resumed = torch.load(classifier_ckpt)
        resumed = torch.load(classifier_ckpt, map_location='cuda:0')
        if 'state_dict' in resumed:
            state_dict_key = 'state_dict'
        else:
            state_dict_key = 'model_state'
        print("Resuming from {}".format(classifier_ckpt))
        safe_load_dict(classifier, resumed[state_dict_key])
    if fe:
        print('Using model as feature extractor')
        for params in classifier.model.parameters():
            params.requires_grad = False
        for params in classifier.model.fc.parameters():
            params.requires_grad = True
    return classifier