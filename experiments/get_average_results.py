import json
import os
import numpy as np
import argparse


def get_results(root, folder, seeds, num_inc, order):
    # initialize arrays for storage
    top1_res = np.zeros((len(seeds), num_inc))
    auroc_res = np.zeros_like(top1_res)
    auosc_res = np.zeros_like(top1_res)
    auosc_norm_res = np.zeros_like(top1_res)

    # put results in main array for each seed
    for e, i in enumerate(seeds):
        with open(os.path.join(folder % i, 'accuracies_index_-1.json')) as file:
            d = json.load(file)

        top1 = d['top1']
        top1_res[e] = top1

        if order == 'class_instance':
            auroc_score_intra = d['auroc_score_intra']
            auroc_res[e] = auroc_score_intra

            auosc_score_intra = d['auosc_score_intra']
            auosc_res[e] = auosc_score_intra

            auosc_norm_score_intra = d['auosc_norm_score_intra']
            auosc_norm_res[e] = auosc_norm_score_intra

    # mean and standard deviation over different runs
    average_top1 = np.mean(top1_res, axis=0)
    std_top1 = np.std(top1_res, axis=0)

    if order == 'class_instance':
        average_auroc = np.mean(auroc_res, axis=0)
        std_auroc = np.std(auroc_res, axis=0)

        average_auosc = np.mean(auosc_res, axis=0)
        std_auosc = np.std(auosc_res, axis=0)

        average_auosc_norm = np.mean(auosc_norm_res, axis=0)
        std_auosc_norm = np.std(auosc_norm_res, axis=0)

        return (average_top1, std_top1), (average_auroc, std_auroc), (average_auosc, std_auosc), (
            average_auosc_norm, std_auosc_norm)
    else:
        return (average_top1, std_top1)


def compute_omega_instance(root, model='slda', num_inc=6, seeds=[10, 20, 30]):

    model_list = [model, 'offline']
    
    results_dict = {}
    for m in model_list:
        folder = os.path.join(root, 'stream51_' + m + '_experiment_instance_seed%d')

        (mu_top1, std_top1) = get_results(root, folder, seeds, num_inc, 'instance')
        results_dict[m] = [mu_top1, std_top1]

    omega_top1 = np.mean(results_dict[model][0][0] / results_dict['offline'][0][0])
    print('\n', model)
    print('\tOmega Top-1: ', omega_top1)


def compute_omega_class_instance(root, model='slda', num_inc=6, seeds=[10, 20, 30]):

    model_list = [model, 'offline']

    results_dict = {}
    for m in model_list:
        folder = os.path.join(root, 'stream51_' + m + '_experiment_class_instance_seed%d')

        (mu_top1, std_top1), (mu_auroc, std_auroc), (mu_auosc, std_auosc), (
            mu_auosc_norm, std_auosc_norm) = get_results(root, folder, seeds, num_inc, 'class_instance')
        results_dict[m] = [[mu_top1, mu_auroc, mu_auosc, mu_auosc_norm],
                           [std_top1, std_auroc, std_auosc, std_auosc_norm]]

    omega_top1 = np.mean(results_dict[model][0][0] / results_dict['offline'][0][0])
    omega_auroc = np.mean(results_dict[model][0][1] / results_dict['offline'][0][1])
    omega_auosc = np.mean(results_dict[model][0][2] / results_dict['offline'][0][2])

    print('\n', model)
    print('\tOmega Top-1: ', omega_top1)
    print('\tOmega AUROC: ', omega_auroc)
    print('\tOmega AUOSC: ', omega_auosc)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default = './results')
    parser.add_argument('--model', type=str, default='slda',
                        choices=['slda', 'exstream', 'no_buffer', 'no_buffer_full'])
    parser.add_argument('--num_inc', type=int, default=6)
    parser.add_argument('--seeds', type=str, default = '10,20,30')
    args = parser.parse_args()
    
    print('Summary Statistics:')
    print('\nInstance Ordering:')
    compute_omega_instance(args.results_dir, args.model, args.num_inc, [int(i) for i in args.seeds.split(',')])

    print('\n\nClass Instance Ordering:')
    compute_omega_class_instance(args.results_dir, args.model, args.num_inc, [int(i) for i in args.seeds.split(',')])

    # compare_offline_ood('instance', root, save_dir, include_std=include_std)
    # compare_offline_ood('class_instance', root, save_dir, include_std=include_std)


if __name__ == '__main__':
    main()
