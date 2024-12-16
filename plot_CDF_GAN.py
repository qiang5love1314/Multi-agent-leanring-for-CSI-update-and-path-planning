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
    x1, y1 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/GAN_Analysis/GAN-Lab-Error.mat')
    x2, y2 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/GAN_Analysis/noGAN-Lab-Error.mat')
    plt.step(x1, y1, color = 'r', marker ='o', label='CPPU with GAN')
    plt.step(x2, y2, color='b', marker='v', label='CPPU without GAN')

    '-----Meet-----'
    # x1, y1 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/GAN_Analysis/GAN-Meet-Error.mat')
    # x2, y2 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/GAN_Analysis/noGAN-Meet-Error.mat')
    # plt.step(x1, y1, color = 'green', marker ='x', label='CPPU with GAN')
    # plt.step(x2, y2, color='c', marker='p', label='CPPU without GAN')
    
    plt.xlabel('Localization Error (m)', size=20)
    plt.ylabel('CDF', size=20)
    plt.grid(color="grey", linestyle=':', linewidth=0.5)
    plt.tick_params(labelsize=13)
    plt.legend(loc = 'lower right', fontsize=15)
    plt.tight_layout()
    # plt.savefig('/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Fig/GAN_Analysis_Lab.pdf', bbox_inches = 'tight')
    plt.show()

if __name__ == '__main__':
    main()
    pass