import numpy as np
from scipy.io import loadmat
import os
import math
import GPy
from sklearn.impute import SimpleImputer
from sklearn.manifold import MDS 
from sklearn.model_selection import train_test_split
from Lab.LabMulAgtPath import find_PotentialPath

class RewardCalculator:
    np.random.seed(123)
    def __init__(self, path="/Users/zhuxiaoqiang/Downloads/47SwapData/"):
        self.path = path

    def get_x_label(self):
        x_label = [str(i + 1) for i in range(21)]  # 横坐标
        return x_label

    def get_y_label(self):
        y_label = [f'{0}{j + 1}' if j < 9 else str(j + 1) for j in range(23)]  # 纵坐标
        return y_label

    def raw_csi(self):
        x_label = self.get_x_label()
        y_label = self.get_y_label()
        count = 0
        original_csi = np.zeros((317, 135000), dtype=np.float)
        labels = np.empty((0, 2), dtype=np.int)

        for i in range(21):
            for j in range(23):
                file_path = os.path.join(self.path, f"coordinate{x_label[i]}{y_label[j]}.mat")
                if os.path.isfile(file_path):
                    c = loadmat(file_path)
                    csi = np.reshape(c['myData'], (1, 3 * 30 * 1500))
                    original_csi[count, :] = csi
                    labels = np.append(labels, [[int(x_label[i]), int(y_label[j])]], axis=0)
                    count += 1
        return original_csi, labels, count

    def SwapValue(self, x):
        max_val = np.max(x)
        min_val = np.min(x)
        k = (5 - 1) / (max_val - min_val)
        value = k * (x - min_val) + 1
        return value

    def get_gaussian(self, values):
        mu = np.mean(values)
        sigma = np.std(values)
        y = (1 / (np.sqrt(2 * np.pi * np.power(sigma, 2)))) * \
            (np.power(np.e, -(np.power((values - mu), 2) / (2 * np.power(sigma, 2)))))
        return y

    def ComputeDifferentialEntropy(self, cov, size):
        hyper_parameter = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                hyper_parameter[i, j] = 0.5 * math.log(abs(cov[i, j]), math.e) + size / 2.0 * \
                                        (1 + math.log(2 * math.pi, math.e))
        return hyper_parameter

    def fit_original_data(self, original):
        original_data = original[:, :3 * 30 * 50].astype(float)
        original_data[np.isnan(original_data)] = 0
        original_data[np.isinf(original_data)] = np.nanmax(original_data)
        imputer = SimpleImputer(copy=False)
        originalData = imputer.fit_transform(original_data)
        return originalData
    
    def calculateRewards(self):
        original, label, count = self.raw_csi()
        originalData = self.fit_original_data(original)
        dimenReduce = MDS(2,random_state=10).fit_transform(originalData)
        
        kernelRBF = GPy.kern.RBF(input_dim=2, variance=1)
        size = len(label)
        mu = np.mean(label, axis=1)
        cov = kernelRBF.K(label, label) 
        H_y = self.ComputeDifferentialEntropy(cov, size)

        traindata, testdata, trainlabel, testlabel = train_test_split(dimenReduce, label, test_size=0.9, random_state=20)
        model = GPy.models.GPRegression(traindata, trainlabel, kernel=kernelRBF)    #计算超参数
        model.optimize()

        gaussian_variance = model.param_array[2]

        part1 = kernelRBF.K(label, trainlabel)
        part2 = np.linalg.inv(kernelRBF.K(trainlabel, trainlabel) + math.pow(gaussian_variance, 2) * np.eye(len(trainlabel)))
        part3 = kernelRBF.K(trainlabel, label)
        covPlus= cov - np.dot(np.dot(part1, part2), part3)      #由少量采集数据计算微分熵
        H_yAnds = self.ComputeDifferentialEntropy(covPlus, size)

        'compute reward based MI'
        reward_MI = H_y - H_yAnds + np.random.normal(loc=0.5, scale=0.1, size=H_yAnds.shape)
        zeroReward = self.SwapValue(np.diag(reward_MI))
        
        xLabel = self.get_x_label()
        yLabel = self.get_y_label()
        count = 0
        diagReward = np.zeros((22, 24), dtype=np.float)
        for i in range(21):
            for j in range(23):
                filePath = f"/Users/zhuxiaoqiang/Downloads/47SwapData/coordinate" + xLabel[i] + yLabel[j] + ".mat"
                if (os.path.isfile(filePath)):
                    swap = zeroReward[count]
                    diagReward[int(xLabel[i]), int(yLabel[j])] = swap
                    count += 1
        return diagReward
    
    # 根据3个agent其可移动范围坐标，分配各自的奖赏值
    def diagReward_selected(self):
        diagReward = np.load('/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Lab/Lab_rewards.npy', allow_pickle=True)  # 已计算直接加载
        coordinateList, sizes = self.get_coordinatesList() 
        diagReward_selected = []
        for robot_coordinates in coordinateList:
            robot_rewards = []
            for coordinate in robot_coordinates:
                x, y = coordinate
                robot_rewards.append(diagReward[x, y])
            diagReward_selected.append(robot_rewards)
        return diagReward_selected

    def get_coordinatesList(self):
        coordinateList = np.array(find_PotentialPath(), dtype=object) # 各机器人可移动范围，array（list1_length=126，list2_length=64，list3_length=127
        sizes = [len(sublist) for sublist in coordinateList]
        return coordinateList, sizes

# reward_calculator = RewardCalculator()
# coordinateList, sizes = reward_calculator.get_coordinatesList()
# print(sizes)