import os
import argparse
import json
import time
import torch
import numpy as np

from models.SLDAModel import StreamingLDA
from models.OutputLayer import Output
from models.ExStream import ExStream
from main_experiment_utils import ResNet18, ModelWrapper, build_classifier, get_stream_data_loader, get_class_change_boundaries, get_ood_loaders, update_and_save_accuracies_ood, save_accuracies

def pool_feat(features):
    feat_size = features.shape[-1]
    num_channels = features.shape[1]
    features2 = features.permute(0, 2, 3, 1)  # 1 x feat_size x feat_size x num_channels
    features3 = torch.reshape(features2, (features.shape[0], feat_size * feat_size, num_channels))
    feat = features3.mean(1)  # mb x num_channels
    return feat


def predict(model, val_data, feature_extraction_wrapper, num_classes, ood_predict=False):
    num_samples = len(val_data.dataset)
    probas = torch.empty(num_samples, num_classes)
    labels = torch.empty(num_samples, dtype=torch.long)
    start = 0
    for X, y in val_data:
        # extract feature from pre-trained model and mean pool
        if feature_extraction_wrapper is not None:
            feat = feature_extraction_wrapper(X.cuda())
            feat = pool_feat(feat)
        else:
            feat = X.cuda()
        end = start + feat.shape[0]

        if ood_predict:
            prob = model.ood_predict(feat, auroc=True)
        else:
            prob = model.predict(feat, return_probas=True)

        probas[start:end] = prob
        labels[start:end] = y.squeeze()
        start = end
    return probas, labels


def run_experiment(dataset, images_dir, order, num_classes, classifier, feature_extraction_model,
                   save_dir, step_size, batch_size=256, augment=False, extraction_layer_name=['model.layer4.1'],
                   ood_num_samples=1000, k_plus_one_tpr=0.95, seed=200, ood_only_51=False):
    start_time = time.time()
    accuracies = {'top1': [], 'top5': [], 'auroc_score_noise': [], 'auosc_score_noise': [],
                  'auosc_norm_score_noise': [], 'k_plus_one_acc_noise': [], 'auroc_score_intra': [],
                  'auosc_score_intra': [], 'auosc_norm_score_intra': [], 'k_plus_one_acc_intra': [], 'time': []}

    if feature_extraction_model is not None:
        feature_extraction_wrapper = ModelWrapper(feature_extraction_model.cuda(),
                                                                                 extraction_layer_name,
                                                                                 return_single=True).eval()
    else:
        feature_extraction_wrapper = None

    tr = get_stream_data_loader(images_dir, True, ordering=order, batch_size=batch_size,
                                  shuffle=False, augment=augment, seed=seed)

    test_loader_full = get_stream_data_loader(images_dir, False, ordering=None,
                                                batch_size=batch_size, seen_classes=None)

    if order in ['iid', 'instance']:
        eval_ix = np.arange(step_size, len(tr.dataset), step_size)
    else:
        train_labels = np.array([t for t in tr.dataset.targets])
        eval_ix = get_class_change_boundaries(train_labels)
        if step_size != 1:
            eval_ix = eval_ix[step_size - 1:-1:step_size - 1]

    # add final point for last evaluation
    eval_ix = np.append(eval_ix, np.array(len(tr.dataset)))
    print('eval_ix ', eval_ix)

    seen_classes = []

    print('Beginning streaming training...')
    i = 0
    for batch_ix, (batch_x, batch_y) in enumerate(tr):

        if feature_extraction_wrapper is not None:
            # extract feature from pre-trained model and mean pool
            batch_x_feat = feature_extraction_wrapper(batch_x.cuda())
            batch_x_feat = pool_feat(batch_x_feat)
        else:
            batch_x_feat = batch_x.cuda()

        for x, y in zip(batch_x_feat, batch_y):

            if i in eval_ix and i != 0:

                print('Making test loader from following: ', seen_classes)

                in_loader, out_loader_noise, out_loader_intra = get_ood_loaders(test_loader_full, num_classes,
                                                                                batch_size, ood_num_samples,
                                                                                seen_classes=seen_classes,
                                                                                dataset=dataset,
                                                                                only_explicit_out_data=ood_only_51)

                if order in ['iid', 'instance']:
                    test_classes = np.arange(num_classes)
                else:
                    test_classes = seen_classes

                
                test_loader = get_stream_data_loader(images_dir, False, ordering=None,
                                                           batch_size=batch_size, seen_classes=test_classes)

                # get classification and novelty detection outputs
                probas, y_test_init = predict(classifier, test_loader, feature_extraction_wrapper, num_classes)
                ood_scores_in, true_labels_in = predict(classifier, in_loader,
                                                        feature_extraction_wrapper, num_classes, ood_predict=True)
                ood_scores_out, true_labels_out = predict(classifier, out_loader_intra,
                                                          feature_extraction_wrapper, num_classes, ood_predict=True)

                update_and_save_accuracies_ood(probas, y_test_init, ood_scores_in, true_labels_in, ood_scores_out,
                                               true_labels_out, save_dir, i, accuracies, num_classes, k_plus_one_tpr)

            # fit model with rehearsal
            classifier.fit(x, y.view(1, ))
            i += 1

            # if class not yet in seen_classes, append it
            if y.item() not in seen_classes:
                seen_classes.append(y.item())

    print('Making test loader from following: ', seen_classes)
    in_loader, out_loader_noise, out_loader_intra = get_ood_loaders(test_loader_full, num_classes,
                                                                    batch_size, ood_num_samples,
                                                                    seen_classes=seen_classes,
                                                                    dataset=dataset,
                                                                    only_explicit_out_data=ood_only_51)


    test_loader = get_stream_data_loader(images_dir, False, ordering=None,
                                           batch_size=batch_size, seen_classes=seen_classes)

    # get classification and novelty detection outputs
    probas, y_test_init = predict(classifier, test_loader, feature_extraction_wrapper, num_classes)
    ood_scores_in, true_labels_in = predict(classifier, in_loader,
                                            feature_extraction_wrapper, num_classes, ood_predict=True)
    ood_scores_out, true_labels_out = predict(classifier, out_loader_intra,
                                              feature_extraction_wrapper, num_classes, ood_predict=True)

    update_and_save_accuracies_ood(probas, y_test_init, ood_scores_in, true_labels_in, ood_scores_out,
                                   true_labels_out, save_dir, i, accuracies, num_classes, k_plus_one_tpr)

    end_time = time.time()
    accuracies['time'].append(end_time - start_time)
    save_accuracies(accuracies, index=-1, save_path=save_dir)
    return accuracies


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='stream51', choices=['stream51', 'icub1', 'core50'])
    parser.add_argument('--images_dir', type=str)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--expt_name', type=str)
    parser.add_argument('--order', type=str, choices=['iid', 'class_iid', 'instance', 'class_instance'])
    parser.add_argument('--ckpt_file', type=str, default=None)
    parser.add_argument('--model', type=str, default='no_buffer_full',
                        choices=['slda', 'exstream', 'no_buffer', 'no_buffer_full'])

    parser.add_argument('--num_classes', type=int)
    parser.add_argument('--num_channels', type=int, default=512)
    parser.add_argument('--ood_num_samples', type=int, default=500)
    parser.add_argument('--slda_ood_type', type=str, default='mahalanobis', choices=['mahalanobis', 'baseline'])
    parser.add_argument('--k_plus_one_tpr', type=float, default=0.95)
    parser.add_argument('--ood_only_51', action='store_true')  # true if we only want class 51 in out-loader

    parser.add_argument('--step', type=int, default=10)
    parser.add_argument('--seed', type=int, default=10)

    args = parser.parse_args()
    print("Arguments {}".format(json.dumps(vars(args), indent=4, sort_keys=True)))

    if args.save_dir is None:
        args.save_dir = 'streaming_experiments/' + args.expt_name
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # define model for feature extraction
    feature_extraction_model = build_classifier(args.ckpt_file, num_classes=args.num_classes).eval()

    model = Output(args.num_channels, args.num_classes, lr=0.01, weight_decay=1e-4, batch_size=256,
                   ckpt_file=args.ckpt_file, resume_ckpt=True)
    if args.model == 'exstream':
        classifier = ExStream(model, args.num_channels, args.num_classes, capacity=100, batch_size=256, buffers='class',
                              lr=0.01, weight_decay=1e-4)
    elif args.model == 'slda':
        classifier = StreamingLDA(args.num_channels, args.num_classes, test_batch_size=512,
                                  shrinkage_param=1e-4, streaming_update_sigma=True,
                                  one_versus_rest=False, ood_type=args.slda_ood_type)
    elif args.model == 'no_buffer':
        classifier = model
    elif args.model == 'no_buffer_full':
        classifier = ResNet18(num_classes=args.num_classes, ckpt=args.ckpt_file)

        # turn batch norm off since we are training one sample at a time
        for module in classifier.modules():
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for param in module.parameters(): param.requires_grad = False
        feature_extraction_model = None
    else:
        raise NotImplementedError('Model not supported.')

    # perform streaming classification
    run_experiment(args.dataset, args.images_dir, args.order, args.num_classes, classifier,
                   feature_extraction_model, args.save_dir, args.step, ood_num_samples=args.ood_num_samples,
                   k_plus_one_tpr=args.k_plus_one_tpr, seed=args.seed, ood_only_51=args.ood_only_51)


if __name__ == '__main__':
    main()
