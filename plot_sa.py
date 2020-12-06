import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import shutil
import json
import numpy as np

sns.set_theme()

fbp_errors = [[0.0, 0.0, 0.2739],
              [0.0, 0.0072, 0.4492],
              [0.0, 0.0054, 0.2756]]
files = ['5494', '5509', '5490']
seeds = [0,1,2,3,4]
thetas = [1,10,30]
iters = [10,100,1000,5000,10000,50000,100000]
temps = [1,5,10,50,100,250,500]

plots_path = "plots"
if not os.path.exists(plots_path):
    os.mkdir(plots_path)

path = "E:\\Egyetem\\képrekonstrukció\\sa\\results"
'''
images = os.listdir(path)
for i in range(len(images)):
    images[i] = os.path.join(path,images[i],'stats')
print(images)

for num,i in enumerate(images):
    stats = glob.glob(i+'\\*.json')
    if not os.path.exists(os.path.join(plots_path,files[num])):
        os.mkdir(os.path.join(plots_path,files[num]))
    json_files = []
    for j in stats:
        for l in temps:
            for k in files:
                sub_path = os.path.join(plots_path,k,'temp-'+str(l))
                if not os.path.exists(sub_path):
                    os.makedirs(sub_path)
                if k in j and 'temp-'+str(l)+'.' in j:
                    shutil.copy(j,sub_path)
'''


img = os.listdir('plots')
for num, i in enumerate(img):
    sub_dirs = os.listdir(os.path.join('plots',i))
    #iter_theta_1 = {'10':[],'100':[],'1000':[],'5000':[],'10000':[],'50000':[],'100000':[]}
    #iter_theta_10 = {'10':[],'100':[],'1000':[],'5000':[],'10000':[],'50000':[],'100000':[]}
    #iter_theta_30 = {'10':[],'100':[],'1000':[],'5000':[],'10000':[],'50000':[],'100000':[]}

    fbp_error = fbp_errors[num]
    for szam, j in enumerate(sub_dirs):
        iter_theta_1 = []
        iter_theta_10 = []
        iter_theta_30 = []
        fi = glob.glob(os.path.join('plots',i,j,'*.json'))
        for it in iters:
            theta_1 = []
            theta_10 = []
            theta_30 = []
            for l in fi:
                for t in thetas:
                    for s in seeds:
                        if 'step-'+str(t)+'_' in l and 'seed-'+str(s)+'_' in l and '_iter-'+str(it)+'_' in l:
                            f = open(l)
                            data = json.load(f)
                            if t == 1:
                                theta_1.append(data["rme"])
                            elif t == 10:
                                theta_10.append(data["rme"])
                            elif t == 30:
                                theta_30.append(data["rme"])
                            f.close()

            iter_theta_1.append(np.mean(theta_1))
            iter_theta_10.append(np.mean(theta_10))
            iter_theta_30.append(np.mean(theta_30))
        labels = ['10','100','1000','5000','10000','50000','100000']

        plt.figure()
        plt.plot(list(range(len(labels))), iter_theta_1, label='SA error - theta 1')
        plt.plot(list(range(len(labels))), iter_theta_10, label='SA error - theta 10')
        plt.plot(list(range(len(labels))), iter_theta_30, label='SA error - theta 30')
        plt.plot(list(range(len(labels))), np.ones(7)*fbp_error[0], label='FBP error - theta 1')
        plt.plot(list(range(len(labels))), np.ones(7)*fbp_error[1], label='FBP error - theta 10')
        plt.plot(list(range(len(labels))), np.ones(7)*fbp_error[2], label='FBP error - theta 30')

        plt.ylabel('RME values')
        plt.xlabel('Iters')
        plt.xticks(np.arange(7), ['10','100','1000','5000','10000','50000','100000'])

        plt.legend(loc='upper right')
        if not os.path.exists('only_plots'):
            os.mkdir('only_plots')
        plt.savefig(os.path.join('only_plots',i+'_results_'+j+'.png'))
