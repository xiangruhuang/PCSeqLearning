import numpy as np
import glob, os
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

name_map = {
    'l1_opt_r025': r'$\lambda=0.25$',
    'l1_opt_r05': r'$\lambda=0.5$',
    'l1_opt_r1': r'$\lambda=1$',
    'l1_opt_r0125': r'$\lambda=0.125$',
    'l1_opt_r006125': r'$\lambda=0.06125$',
}

folders = sorted([f for f in glob.glob('../output/waymo_sequence_registration/cluster_proposal/*')])

subfolder_dict = defaultdict(list)
plot_dict = {}
for folder in folders:
    # algorithm
    plot_curves = {i: defaultdict(lambda: 0) for i in range(1, 4)}
    seq_txt_files = sorted(glob.glob(f'{folder}/*.txt'))
    algorithm = folder.split('/')[-1]
    plot_dict[algorithm] = defaultdict(list)
    ious = []
    semantics = []
    #seq = [float(h.split('/')[-1][6:]) for h in heightfolders]
    for seq_txt_file in tqdm(seq_txt_files):
        with open(seq_txt_file, 'r') as fin:
            lines = [line.strip() for line in fin.readlines()]
            for line in lines:
                semantic=round(float(line.split('=')[1].split(',')[0]))
                iou = float(line.split('=')[-1])
                ious.append(iou)
                semantics.append(semantic)
    ious = np.array(ious)
    semantics = np.array(semantics)
    print(f"algorithm={algorithm}")
    print(f"Overall={ious.shape[0]}")
    for semantic_label in range(1, 4):
        print(f'semantic_label={semantic_label}')
        for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            print(thresh, (ious[semantics == semantic_label] > thresh).sum())

#for key, string in [
#        ['ground_vs_foreground', 'Ground Removal Truncate Height Trade-Off'],
#        #['#removed_points', 'Removed Points'],
#        #['#removed_foreground', 'Removed ForeGround Points'],
#        #['#removed_ground', 'Removed Ground Points'],
#        #['ground_precision', 'Removed Ground / Removed Points'],
#        #['ground_coverage', 'Removed Ground / Ground Points'],
#        #['foreground_precision', 'Removed ForeGround / Removed Points'],
#        #['foreground_coverage', 'Removed ForeGround / ForeGround Points'],
#    ]:
#    fig = plt.figure()
#    plt.title(string, fontsize=20)
#    plt.ylabel('Coverage of Foreground Points', fontsize=20)
#    plt.xlabel('Coverage of Ground Points', fontsize=20)
#    for algorithm in plot_dict.keys():
#        plt.plot(plot_dict[algorithm][key][:, 0], plot_dict[algorithm][key][:, 1], label=name_map[algorithm])
#    plt.legend(fontsize=20)
#    plt.savefig(f"{key.replace('#', '')}.ps")
#    os.system(f"ps2eps -f {key.replace('#', '')}.ps")
#    os.system(f"rm {key.replace('#', '')}.ps")
#    #plt.show()
