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
    # x1, y1 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/PSRO_Path_Analysis/LabLocalization/10iter-Lab-Error.mat')
    # x2, y2 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/PSRO_Path_Analysis/LabLocalization/20iter-Lab-Error.mat')
    # x3, y3 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/PSRO_Path_Analysis/LabLocalization/30iter-Lab-Error.mat')
    # x4, y4 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/PSRO_Path_Analysis/LabLocalization/40iter-Lab-Error.mat')
    # x5, y5 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/PSRO_Path_Analysis/LabLocalization/50iter-Lab-Error.mat')
    
    '-----Meet-----'
    x1, y1 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/PSRO_Path_Analysis/MeetLocalization/5iter-Meet-Error.mat')
    x2, y2 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/PSRO_Path_Analysis/MeetLocalization/10iter-Meet-Error.mat')
    x3, y3 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/PSRO_Path_Analysis/MeetLocalization/15iter-Meet-Error.mat')
    x4, y4 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/PSRO_Path_Analysis/MeetLocalization/20iter-Meet-Error.mat')
    x5, y5 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/PSRO_Path_Analysis/MeetLocalization/25iter-Meet-Error.mat')
   
    # sample = loadmat(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/AgentNum_Analysis/Meet_2Agents_5iter-Error.mat')
    # sample = np.array(sample['array']).flatten()
    # print(np.mean(sample))
    # print(np.std(sample))

    plt.step(x1, y1, color = 'r', marker ='o', label='iter=10')
    plt.step(x2, y2, color='b', marker='v', label='iter=20')
    plt.step(x3, y3, color='green', marker='x', label='iter=30')
    plt.step(x4, y4, color='c', marker='p', label='iter=40')
    plt.step(x5, y5, color = 'orange', marker = '*', label = 'iter=50')

    plt.xlabel('Localization Error (m)', size=20)
    plt.ylabel('CDF', size=20)
    plt.grid(color="grey", linestyle=':', linewidth=0.5)
    plt.tick_params(labelsize=13)
    plt.legend(loc = 'lower right', fontsize=15)
    plt.tight_layout()
    # plt.savefig('/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Fig/iter_PSRO_Meet.pdf', bbox_inches = 'tight')
    plt.show()

if __name__ == '__main__':
    main()
    pass