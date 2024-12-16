import statsmodels.api as sm
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

def read(path):
    sample = loadmat(path)
    sample = np.array(sample['array']).flatten()
    # print(np.mean(sample))
    ecdf = sm.distributions.ECDF(sample)
    x = np.linspace(min(sample), max(sample))
    y = ecdf(x)
    return x, y

def main():
    '-----Lab-----'
    x1, y1 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/CPPU-Lab-Error.mat')
    x2, y2 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/Original-Lab-Error.mat')
    x3, y3 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/TBD-Lab-Error.mat')
    x4, y4 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/A3C-Lab-Error.mat')
    x5, y5 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/QLearning-Lab-Error.mat')
    x6, y6 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/INFOCOM2020-Lab-Error.mat')

    'random strategy for reviewer'
    # x1, y1 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/CPPU-Lab-Error.mat')
    # x2, y2 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/RandomSampling_Analysis/Lab_percent_10-Error.mat')
    # x3, y3 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/RandomSampling_Analysis/Lab_percent_30-Error.mat')
    # x4, y4 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/RandomSampling_Analysis/Lab_percent_50-Error.mat')

    '-----Meet-----'
    # x1, y1 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/CPPU-Meet-Error.mat')
    # x2, y2 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/Original-Meet-Error.mat')
    # x3, y3 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/TBD-Meet-Error.mat')
    # x4, y4 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/A3C-Meet-Error.mat')
    # x5, y5 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/QLearning-Meet-Error.mat')
    # x6, y6 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/INFOCOM2020-Meet-Error.mat')
    
    'random strategy for reviewer'
    # x1, y1 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/CPPU-Meet-Error.mat')
    # x2, y2 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/RandomSampling_Analysis/Meet_percent_10-Error.mat')
    # x3, y3 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/RandomSampling_Analysis/Meet_percent_30-Error.mat')
    # x4, y4 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/RandomSampling_Analysis/Meet_percent_50-Error.mat')
    
    # plt.step(x1, y1, color = 'r', marker ='o', label='CPPU')
    # plt.step(x2, y2, color='b', marker='v', label='10% Random Sampling')
    # plt.step(x3, y3, color='green', marker='x', label='30% Random Sampling')
    # plt.step(x4, y4, color='c', marker='p', label='50% Random Sampling')

    # sample = loadmat(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/RandomSampling_Analysis/Lab_percent_50-Error.mat')
    # sample = np.array(sample['array']).flatten()
    # print(np.mean(sample))
    # print(np.std(sample))

    plt.step(x1, y1, color = 'r', marker ='o', label='CPPU')
    plt.step(x2, y2, color='b', marker='v', label='Original Database')
    plt.step(x3, y3, color='green', marker='x', label='ILUC')
    plt.step(x4, y4, color='c', marker='p', label='A3C-IPP')
    plt.step(x5, y5, color = 'orange', marker = '*', label = 'DQN-IPP')
    plt.step(x6, y6, color = 'y', marker = '+', label = 'GA-IPP')

    plt.xlabel('Localization Error (m)', size=20)
    plt.ylabel('CDF', size=20)
    plt.grid(color="grey", linestyle=':', linewidth=0.5)
    plt.tick_params(labelsize=13)
    plt.legend(loc = 'lower right', fontsize=13)
    plt.tight_layout()
    # plt.savefig('/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Fig/ComDiffAlgoAccuracy_Lab.pdf', bbox_inches = 'tight')
    # plt.savefig('/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Fig/RandomSampling_Meet.pdf', bbox_inches = 'tight')
    plt.show()

if __name__ == '__main__':
    main()
    pass