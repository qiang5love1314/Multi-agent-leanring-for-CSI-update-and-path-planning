import numpy as np
import matplotlib.pyplot as plt

def computeFinalReward():
    '----lab----'
    initialReward = np.load('/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Lab/Lab_rewards.npy', allow_pickle=True)  # 已计算直接加载
    finalPath = np.load('/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/RewardCompare/lab/5Agents_Optimal50iter_final_path.npy', allow_pickle=True)
    
    '----meeting room----'
    # initialReward = np.load('/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/MeetingRoom/Meet_rewards.npy', allow_pickle=True)  # 已计算直接加载
    # finalPath = np.load('/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/RewardCompare/meet/Meet_5Agents_Optimal25iter_final_path.npy', allow_pickle=True)
    # print(sum(len(sublist) for sublist in finalPath))
    diagReward_selected = []
    for robot_coordinates in finalPath:
        robot_rewards = []
        for coordinate in robot_coordinates:
            x, y = coordinate
            robot_rewards.append(initialReward[x, y])
        diagReward_selected.append(robot_rewards)
    
    reward = diagReward_selected
    TotalReward = 0
    for sublist in reward:
        for item in sublist:
            TotalReward += item
    return round(TotalReward,2)

def main():
    figure, ax = plt.subplots()

    '----------Lab-----------'
    # stepLength = ['10', '20', '30', '40', '50']
    # twoAgents = [151.10, 211.92, 286.50, 318.68, 358.35]
    # threeAgents = [142.88, 284.40, 330.32, 375.27, 399.06]
    # fourAgents = [200.64, 316.52, 362.73, 395.97, 414.97]
    # fiveAgents = [ 193.09, 324.38, 389.28, 414.54, 426.10]

    '----------Meeting Room-----------'
    stepLength = ['5', '10', '15', '20', '25']
    twoAgents = [42.39, 104.41, 161.50, 192.42, 216.45]
    threeAgents = [63.07, 135.05, 156.06, 210.52, 233.31]
    fourAgents = [93.96, 138.64, 184.7, 203.00, 224.59]
    fiveAgents = [85.7, 140.35, 191.81, 205.67, 227.77]

    plt.plot(stepLength, twoAgents, color='r', marker='o', label='Two Agents')
    plt.plot(stepLength, threeAgents, color='b', marker='v', label='Three Agents')
    plt.plot(stepLength, fourAgents, color='g', marker='x', label='Four Agents')
    plt.plot(stepLength, fiveAgents, color='c', marker='p', label='Five Agents')

    plt.xlabel('Iteration Numbers', size=18)
    plt.ylabel('Total Rewards (MI)', size=18)
    plt.grid(color="grey", linestyle=':', linewidth=0.5)
    plt.tick_params(labelsize=13)
    plt.legend(loc='lower right', fontsize=15)
    plt.savefig('/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Fig/rewardCompare_meet.pdf', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    # print(computeFinalReward())
    main()