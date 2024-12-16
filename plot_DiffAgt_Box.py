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

x1, y1 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/AgentNum_Analysis/2Agents_10iter-Lab-Error.mat')
x2, y2 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/AgentNum_Analysis/3Agents_10iter-Lab-Error.mat')
x3, y3 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/AgentNum_Analysis/4Agents_10iter-Lab-Error.mat')
x4, y4 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/AgentNum_Analysis/5Agents_10iter-Lab-Error.mat')

# x1, y1 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/AgentNum_Analysis/Meet_2Agents_5iter-Error.mat')
# x2, y2 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/AgentNum_Analysis/Meet_3Agents_5iter-Error.mat')
# x3, y3 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/AgentNum_Analysis/Meet_4Agents_5iter-Error.mat')
# x4, y4 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/AgentNum_Analysis/Meet_5Agents_5iter-Error.mat')

data = [x1, x2, x3, x4]
colors = ['r', 'b', 'g', 'c']

# 画箱线图，使用 patch_artist=True 以便填充颜色
box = plt.boxplot(data, patch_artist=True)

# 给每个箱线图填充颜色
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

median_color = 'black'
for median in box['medians']:
    median.set_color(median_color)

# plt.boxplot(data)
plt.xticks([1, 2, 3, 4], ['2', '3', '4', '5'])

plt.xlabel('The Number of Agents', size=20)
plt.ylabel('Localization Error (m)', size=20)
plt.grid(color="grey", linestyle=':', linewidth=0.5)
plt.tick_params(labelsize=13)
plt.tight_layout()
plt.savefig('/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Fig/DiffAgent_Lab.pdf', bbox_inches = 'tight')  
plt.show()
