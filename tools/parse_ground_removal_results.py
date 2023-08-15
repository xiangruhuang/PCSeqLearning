import numpy as np
import glob, os
from collections import defaultdict
from tqdm import tqdm

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

name_map = {
    #'_l1_opt_r025': r'$\lambda=0.25$',
    #'_l1_opt_r05': r'$\lambda=0.5$',
    #'l1_opt_r0x5_TLS': r'$\lambda=0.5 + TLS$',
    'l1_opt_r0x5_TLS_k8': r'$\lambda=0.5 + TLS$',
    #'l1_opt_r0x5_dynamic': r'$\lambda=0.5 + dynamic$',
    'l1_opt_r0x5_dynamic_ransac': r'$\lambda=0.5 + dynamic ransac$',
    #'l1_opt_r0375': r'$\lambda=0.375$',
    #'l1_opt_r05_new': r'$\lambda=0.5$ Repeat',
    #'l1_opt_r1': r'$\lambda=1$',
    #'l1_opt_r1_long': r'$\lambda=1$, iter=4000',
    #'_l1_opt_r0125': r'$\lambda=0.125$',
    #'l1_opt_r006125': r'$\lambda=0.06125$',
}

folders = [f for f in glob.glob('../output/waymo_sequence_registration/ground_removal/*')]
subfolder_dict = defaultdict(list)
plot_dict = {}
seq_dict = defaultdict(dict)
for folder in folders:
    algo_name = folder.split('/')[-1]
    if algo_name not in name_map:
        continue
    heightfolders = glob.glob(f'{folder}/log/*')
    algorithm = folder.split('/')[-1]
    plot_dict[algorithm] = defaultdict(list)
    avg_dict = defaultdict(list)
    median_dict = defaultdict(list)
    heightfolders = sorted(heightfolders, key=lambda s: float(s.split('/')[-1][6:]))
    heights = [float(h.split('/')[-1][6:]) for h in heightfolders]
    for height, heightfolder in tqdm(zip(heights, heightfolders)):
        res_dict = defaultdict(list)
        txt_files = glob.glob(f'{heightfolder}/*.txt')
        for txt_file in txt_files:
            sequence_id = txt_file.split('/')[-1].split('.')[0]
            seq_res_dict = dict()

            with open(f'{txt_file}', 'r') as fin:
                lines = [line.strip() for line in fin.readlines()][1:]

            for line in lines:
                left, right = line.split('=')
                right = float(right)
                if left in ['ground_coverage', 'foreground_coverage']:
                    seq_res_dict[left] = right
                res_dict[left].append(right)
            seq_dict[sequence_id][algo_name] = [seq_res_dict['ground_coverage'],
                                                seq_res_dict['foreground_coverage'],
                                                height]
        #avg_dict['ground_vs_foreground'].append([np.mean(res_dict['ground_coverage']), np.mean(res_dict['foreground_coverage'])])
        #for key in res_dict.keys():
        #    avg_dict[key].append([height, np.mean(res_dict[key])])
        #median_dict['ground_vs_foreground'].append([np.median(res_dict['ground_coverage']), np.median(res_dict['foreground_coverage'])])
        #for key in res_dict.keys():
        #    median_dict[key].append([height, np.median(res_dict[key])])

    #for key, val in avg_dict.items():
    #    plot_dict[algorithm]['avg_'+key] = np.array(val)
    #for key, val in median_dict.items():
    #    plot_dict[algorithm]['median_'+key] = np.array(val)

with open('ground_removal_results/compare.txt', 'w') as fout:
    for seq_id, seq_res_dict in seq_dict.items():
        fout.write(f"{seq_id}:\n")
        for algo_name, algo_res in seq_res_dict.items():
            #if algo_res[0] > 0.98 and algo_res[1] < 0.10:
            fout.write(f"\t {algo_name}(h={algo_res[2]:.2f}): GroundCoverage={algo_res[0]:.4f}, ForeGroundCoverage={algo_res[1]:.4f}\n")
    
with open('ground_removal_results/good_sequences.txt', 'w') as good:
    for seq_id, seq_res_dict in seq_dict.items():
        for algo_name in name_map.keys():
            if algo_name not in seq_res_dict:
                continue
            algo_res = seq_res_dict[algo_name]
            if algo_res[0] > 0.97 and algo_res[1] < 0.20:
                good.write(f"{seq_id} @{algo_name}(h={algo_res[2]:.2f}): GroundCoverage={algo_res[0]:.4f}, ForeGroundCoverage={algo_res[1]:.4f}\n")
                break

#for key, string in [
#        #['ground_vs_foreground', 'Ground Removal Truncate Height Trade-Off'],
#        #['#removed_points', 'Removed Points'],
#        #['#removed_foreground', 'Removed ForeGround Points'],
#        #['#removed_ground', 'Removed Ground Points'],
#        ['avg_ground_precision', 'Mean(Removed Ground / Removed Points)'],
#        ['avg_ground_coverage', 'Mean(Removed Ground / Ground Points)'],
#        ['avg_foreground_precision', 'Mean(Removed ForeGround / Removed Points)'],
#        ['avg_foreground_coverage', 'Mean(Removed ForeGround / ForeGround Points)'],
#        ['median_ground_precision', 'Median(Removed Ground / Removed Points)'],
#        ['median_ground_coverage', 'Median(Removed Ground / Ground Points)'],
#        ['median_foreground_precision', 'Median(Removed ForeGround / Removed Points)'],
#        ['median_foreground_coverage', 'Median(Removed ForeGround / ForeGround Points)'],
#    ]:
#    fig = plt.figure()
#    plt.title(string, fontsize=20)
#    plt.ylabel(string, fontsize=20)
#    plt.xlabel('Height', fontsize=20)
#    for algorithm in plot_dict.keys():
#        if algorithm in name_map:
#            print(algorithm, key, plot_dict[algorithm][key])
#            if len(plot_dict[algorithm][key]) > 0:
#                plt.plot(plot_dict[algorithm][key][:, 0], plot_dict[algorithm][key][:, 1], label=name_map[algorithm])
#    plt.legend(fontsize=10)
#    plt.savefig(f"figures/{key.replace('#', '')}.ps")
#    os.system(f"ps2eps -f figures/{key.replace('#', '')}.ps")
#    os.system(f"rm figures/{key.replace('#', '')}.ps")
#    #plt.show()
