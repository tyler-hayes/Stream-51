import torch
from torch import nn
import os


class StreamingLDA(nn.Module):
    """
    This is an implementation of the Streaming Linear Discriminant Analysis algorithm for streaming learning.
    """

    def __init__(self, input_shape, num_classes, test_batch_size=1024, shrinkage_param=1e-4,
                 streaming_update_sigma=True, one_versus_rest=True, means=None, Sigma=None, num_updates=0, ood_fpr=0.5,
                 ood_type='mahalanobis'):
        """
        Init function for the SLDA model.
        :param input_shape: number of dimensions of features
        :param num_classes: number of total classes in stream
        :param test_batch_size: batch size for inference
        :param shrinkage_param: value of the shrinkage parameter
        :param streaming_update_sigma: True if sigma is plastic else False
        :param one_versus_rest: True if using one-versus-rest formulation else False
        :param Sigma: optional initialized covariance matrix
        :param num_updates: optional number of updates o finitialized covariance matrix
        """

        super(StreamingLDA, self).__init__()

        # SLDA parameters
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cuda'
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.test_batch_size = test_batch_size
        self.shrinkage_param = shrinkage_param
        self.streaming_update_sigma = streaming_update_sigma
        self.one_versus_rest = one_versus_rest
        self.ood_fpr = ood_fpr
        # setup weights for SLDA

        if means is None:
            self.posW = torch.zeros((num_classes, input_shape)).to(self.device)
            self.negW = torch.zeros((num_classes, input_shape)).to(self.device)
            self.posT = torch.zeros(num_classes).to(self.device)
            self.negT = torch.zeros(num_classes).to(self.device)
        else:
            self.posW = means.to(self.device).float()
            self.negW = torch.zeros((num_classes, input_shape)).to(self.device)
            self.posT = torch.zeros(num_classes).to(self.device)
            self.negT = torch.zeros(num_classes).to(self.device)

        self.ood_type = ood_type

        if one_versus_rest:
            self.score_threshold = torch.zeros(num_classes)
        else:
            self.score_threshold = 0.

        # make initial covariance matrix, or use covariance matrix provided
        if Sigma is None:
            self.Sigma = torch.ones((input_shape, input_shape)).to(self.device)
        else:
            self.Sigma = Sigma.to(self.device).float()
        self.num_updates = num_updates

    def fit(self, x, y):
        """
        Fit the SLDA model to a new sample (x,y).
        :param x: a torch tensor of the input data (must be a vector)
        :param y: a torch tensor of the input label
        :return: None
        """
        x = x.to(self.device)
        y = y.long().to(self.device)

        # make sure things are the right shape
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        if len(y.shape) == 0:
            y = y.unsqueeze(0)

        with torch.no_grad():

            # covariance updates
            if self.streaming_update_sigma:
                x_minus_mu = (x - self.posW[y])
                mult = torch.matmul(x_minus_mu.transpose(1, 0), x_minus_mu)
                delta = mult * self.num_updates / (self.num_updates + 1)
                self.Sigma = (self.num_updates * self.Sigma + delta) / (self.num_updates + 1)

            # update positive and negative class means
            self.posW[y, :] += (x - self.posW[y, :]) / (self.posT[y] + 1).unsqueeze(1)
            self.posT[y] += 1
            neg_mask = torch.ones(self.num_classes, dtype=torch.bool)
            neg_mask[y] = 0
            self.negW[neg_mask, :] += (x - self.negW[neg_mask, :]) / (self.negT[neg_mask] + 1).unsqueeze(1)
            self.negT[neg_mask] += 1

            self.num_updates += 1

    def set_slda_parameters(self, d):
        """
        Grab the appropriate model parameters for testing.
        :param d: None or a dictionary containing the model parameters
        :return: model parameters to be used for testing
        """
        if d is None:
            return self.posW, self.negW, self.posT, self.negT, self.Sigma
        else:
            return d['posW'], d['negW'], d['posT'], d['negT'], d['Sigma']

    def predict(self, X, return_probas=False, save_file=None):
        """
        Make predictions on test data X.
        :param X: a torch tensor that contains N data samples (N x d)
        :param return_probas: True if the user would like probabilities instead of predictions returns
        :return: the test predictions or probabilities
        """
        X = X.to(self.device)

        posW, negW, _, _, Sigma = self.set_slda_parameters(save_file)

        with torch.no_grad():
            # initialize parameters for testing
            num_samples = X.shape[0]
            scores = torch.empty((num_samples, self.num_classes))
            mb = min(self.test_batch_size, num_samples)
            # parameters for predictions
            invC = torch.pinverse(
                (1 - self.shrinkage_param) * Sigma + self.shrinkage_param * torch.eye(self.input_shape).to(
                    self.device))
            M = posW.transpose(1, 0)
            if self.one_versus_rest:
                # parameters for predictions
                notM = negW.transpose(1, 0)
                W = torch.matmul(invC, (M - notM))
                invCnotM = torch.matmul(invC, notM)
                invCM = torch.matmul(invC, M)
                c = 0.5 * (- torch.sum(notM * invCnotM, dim=0) + torch.sum(M * invCM, dim=0))
            else:
                W = torch.matmul(invC, M)
                c = 0.5 * torch.sum(M * W, dim=0)

            # loop in mini-batches over test samples
            for i in range(0, num_samples, mb):
                start = min(i, num_samples - mb)
                end = i + mb
                x = X[start:end]
                scores[start:end, :] = torch.matmul(x, W) - c

            # return predictions or probabilities
            if not return_probas:
                return scores.cpu()
            else:
                return torch.softmax(scores, dim=1).cpu()

    def fit_base(self, X, y):
        """
        Fit the SLDA model to the base data.
        :param X: an Nxd torch tensor of base initialization data
        :param y: an Nx1-dimensional torch tensor of the associated labels for X
        :return: None
        """
        print('\nFitting Base...')
        X = X.to(self.device)
        y = y.squeeze()

        # update positive and negative means
        cls_ix = torch.arange(self.num_classes)
        for k in torch.unique(y):
            self.posW[k] = X[y == k].mean(0)
            self.posT[k] = X[y == k].shape[0]
        for k in cls_ix:
            self.negW[k] = X[y != k].mean(0)
            self.negT[k] = X[y != k].shape[0]
        self.num_updates = X.shape[0]

        print('\nEstimating initial covariance matrix...')
        from sklearn.covariance import OAS
        cov_estimator = OAS(assume_centered=True)
        cov_estimator.fit((X - self.posW[y]).cpu().numpy())
        self.Sigma = torch.from_numpy(cov_estimator.covariance_).float().to(self.device)

        print('\nBuilding initial OOD threshold(s)...')
        self.ood_predict(X, y)

        print('')

    def ood_predict(self, X, y=[], auroc=True):
        def pd_mat(input1, input2, precision):  # assumes diagonal precision (kxd)
            f1 = (input1[:, None] - input2)
            f2 = f1.matmul(precision[None, :, :])
            return 0.5 * torch.diagonal(f2.matmul(f1.transpose(2, 1)), dim1=1, dim2=2)

        X = X.to(self.device)

        invC = torch.pinverse(
            (1 - self.shrinkage_param) * self.Sigma + self.shrinkage_param * torch.eye(self.input_shape).to(
                self.device))

        num_samples = X.shape[0]
        mb = min(self.test_batch_size, num_samples)
        scores = torch.empty((num_samples, self.num_classes))
        for i in range(0, num_samples, mb):
            start = min(i, num_samples - mb)
            end = i + mb
            x = X[start:end]

            if self.ood_type == 'mahalanobis':
                scores[start:end, :] = -pd_mat(x, self.posW, invC)
            elif self.ood_type == 'baseline':
                scores[start:end, :] = self.predict(x, return_probas=True)
            else:
                raise NotImplementedError

        if list(y):  # initialization / update of thresholds
            if len(y.shape) > 1:
                y = y.squeeze()
            scores = scores[torch.arange(num_samples), y]
            if self.one_versus_rest:
                for k in torch.unique(y):
                    class_scores = scores[k == y]
                    self.score_threshold[k] = torch.sort(class_scores)[0][int((1 - self.ood_fpr) * len(class_scores))]
            else:
                self.score_threshold = torch.sort(scores)[0][int((1 - self.ood_fpr) * scores.shape[0])]
        else:  # prediction
            if auroc:
                # return scores to compute AUROC
                return scores
            else:
                # return thresholded scores for F1 Score
                scores, preds = scores.max(dim=1)
                if self.one_versus_rest:
                    return scores > self.score_threshold[preds]
                else:
                    return scores > self.score_threshold

    def save_model(self, save_path, save_name):
        """
        Save the model parameters to a torch file.
        :param save_path: the path where the model will be saved
        :param save_name: the name for the saved file
        :return:
        """
        # grab parameters for saving
        d = dict()
        d['posW'] = self.posW.cpu()
        d['negW'] = self.negW.cpu()
        d['posT'] = self.posT.cpu()
        d['negT'] = self.negT.cpu()
        d['Sigma'] = self.Sigma.cpu()
        d['num_updates'] = self.num_updates

        # save model out
        torch.save(d, os.path.join(save_path, save_name + '.pth'))
