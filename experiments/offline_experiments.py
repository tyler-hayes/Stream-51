import os
import argparse
import json
import time
import numpy as np

from main_experiment_utils import get_stream_data_loader, get_class_change_boundaries, get_ood_loaders, update_and_save_accuracies_ood, save_accuracies
from models.ResNet18_Offline import ResNet18_Offline

def run_experiment(args, dataset, images_dir, order, num_classes, save_dir, step_size,
                   full_rehearsal=False, augment=False, ood_num_samples=1000, k_plus_one_tpr=0.95, seed=200,
                   test_batch_size=512):
    batch_size = args.batch_size
    start_time = time.time()
    accuracies = {'top1': [], 'top5': [], 'auroc_score_intra': [], 'auosc_score_intra': [],
                  'auosc_norm_score_intra': [], 'k_plus_one_acc_intra': [], 'time': []}

    tr = get_stream_data_loader(images_dir, True, ordering=order,
                                  batch_size=batch_size,
                                  shuffle=False, augment=augment, seed=seed)

    test_loader_full = get_stream_data_loader(images_dir, False, ordering=None,
                                                batch_size=test_batch_size, seen_classes=None)

    if order in ['iid', 'instance']:
        eval_ix = np.arange(step_size, len(tr.dataset), step_size)
    else:
        train_labels = np.array([t for t in tr.dataset.targets])
        eval_ix = get_class_change_boundaries(train_labels)
        if step_size != 1:
            eval_ix = eval_ix[step_size - 1::step_size]

    # add final point for last evaluation
    eval_ix = np.append(eval_ix, np.array(len(tr.dataset)))
    # eval_ix = [eval_ix[-1]]  # TODO: uncomment this line when doing grid search
    print('eval_ix ', eval_ix)

    for cc, i in enumerate(eval_ix):
        print('\n\nTraining on samples: 0 - %d.' % i)

        if cc == 0 or not full_rehearsal:
            classifier = ResNet18_Offline(num_classes, classifier_ckpt=args.ckpt_file,
                                          imagenet_weights=args.use_imagenet_weights,
                                          feature_extractor=args.use_feature_extractor,
                                          batch_size=args.batch_size, init_lr=args.init_lr, milestones=args.milestones,
                                          num_epochs=args.num_epochs, weight_decay=args.weight_decay, save_dir=save_dir,
                                          save_ckpt=True)

        # indices to train from (subset of dataset)
        ix = np.arange(i)

        # update seen classes
        tr_lab = []
        for jj, t in enumerate(tr.dataset.targets):
            if jj == i:
                break
            if t not in tr_lab:
                tr_lab.append(t)
        seen_classes = tr_lab

        print('\nSeen Classes: ', seen_classes)

        if order in ['iid', 'instance']:
            test_classes = np.arange(num_classes)
        else:
            test_classes = seen_classes

        # train data
        train_loader = get_stream_data_loader(images_dir, True, ordering=order,
                                                batch_size=batch_size, shuffle=True, augment=augment, seed=seed,
                                                ix=ix)
        test_loader = get_stream_data_loader(images_dir, False, ordering=None,
                                               batch_size=test_batch_size, seen_classes=test_classes)
        # test data novelty detection
        in_loader, _, out_loader_intra = get_ood_loaders(test_loader_full, num_classes,
                                                         test_batch_size, ood_num_samples,
                                                         seen_classes=seen_classes, dataset=dataset,
                                                         only_explicit_out_data=args.ood_only_51)

        # fit model
        classifier.fit(train_loader, test_loader, i)

        # get classification and novelty detection outputs
        _, probas, y_test_init = classifier.predict(test_loader)
        _, ood_scores_in, true_labels_in = classifier.predict(in_loader)
        _, ood_scores_out, true_labels_out = classifier.predict(out_loader_intra)

        update_and_save_accuracies_ood(probas, y_test_init, ood_scores_in, true_labels_in, ood_scores_out,
                                       true_labels_out, save_dir, i, accuracies, num_classes, k_plus_one_tpr)

    end_time = time.time()
    accuracies['time'].append(end_time - start_time)
    save_accuracies(accuracies, index=-1, save_path=save_dir)
    return accuracies


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='stream51', choices=['stream51', 'icub1', 'core50'])
    parser.add_argument('--images_dir', type=str, default='/home/ryne/Research/datasets/Stream-51/')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--expt_name', type=str, default='test')
    parser.add_argument('--order', type=str, choices=['iid', 'class_iid', 'instance', 'class_instance'], default='class_instance')
    parser.add_argument('--ckpt_file', type=str, default=None)
    parser.add_argument('--full_rehearsal', action='store_true')

    parser.add_argument('--num_classes', type=int, default=51)
    parser.add_argument('--num_channels', type=int, default=512)
    parser.add_argument('--ood_num_samples', type=int, default=500)
    parser.add_argument('--k_plus_one_tpr', type=float, default=0.95)
    parser.add_argument('--ood_only_51', action='store_true')  # true if we only want class 51 in out-loader

    parser.add_argument('--step', type=int, default=10)
    parser.add_argument('--seed', type=int, default=10)

    # resnet parameters
    parser.add_argument('--use_imagenet_weights', action='store_true')
    parser.add_argument('--use_feature_extractor', action='store_true')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--init_lr', type=float, default=0.1)
    parser.add_argument('--milestones', type=int, nargs='+', default=[15, 30])
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--augment', action='store_true')

    args = parser.parse_args()
    print("Arguments {}".format(json.dumps(vars(args), indent=4, sort_keys=True)))

    if args.save_dir is None:
        
        args.save_dir = 'streaming_experiments/' + args.expt_name
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # perform streaming classification
    run_experiment(args, args.dataset, args.images_dir, args.order, args.num_classes, args.save_dir,
                   args.step, full_rehearsal=args.full_rehearsal, ood_num_samples=args.ood_num_samples,
                   k_plus_one_tpr=args.k_plus_one_tpr, seed=args.seed, augment=args.augment)


if __name__ == '__main__':
    main()
