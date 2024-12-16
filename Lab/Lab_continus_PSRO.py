import numpy as np
from Lab.Lab_PSRO import coordinateList
import matplotlib.pyplot as plt

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def move_towards(current, end, step=1):
    direction = np.array(end) - np.array(current)
    direction = direction / np.linalg.norm(direction)
    next_position = np.round(current + direction * step).astype(int)
    return next_position

def generate_intermediate_points(start, end, step=1):
    points = []
    current = np.array(start)
    end = np.array(end)
    while distance(current, end) > step:
        next_position = move_towards(current, end, step)
        if (next_position == current).all():
            break
        points.append(next_position.tolist())
        current = next_position
    return points

def is_within_region(point, region):
    return tuple(point) in region

def connect_points(points, coordinate_list, step=1):
    connected_points = [points[0]]
    for i in range(1, len(points)):
        previous_point = connected_points[-1]
        current_point = points[i]
        if distance(previous_point, current_point) > step:
            intermediate_points = generate_intermediate_points(previous_point, current_point, step)
            for p in intermediate_points:
                if is_within_region(p, coordinate_list) and p not in connected_points:
                    connected_points.append(p)
        if current_point not in connected_points:  # 去重
            connected_points.append(current_point)
    return connected_points

def findFinalPath():
    # results = np.load('optimal_path_CPPU.npy', allow_pickle=True)
    # results = np.load('/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/PSRO_Path_Analysis/Lab_iter50_PSRO_Path.npy', allow_pickle=True)
    # results = np.load('/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/AgentNum_Analysis/Lab/Optimal50iter_ThreeAgent_(1,1)(12,1)(21,1)_path.npy', allow_pickle=True)
    # results = np.load('/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/AgentNum_Analysis/2Agents_Optimal10iter_path.npy', allow_pickle=True)
    results = np.load('/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/RewardCompare/lab/5Agents_Optimal50iter_path.npy', allow_pickle=True)
    results = results.tolist()

    continuous_results = []
    for agent_idx in range(5):  # 4是agents数量 需更改
        agent_results = [results[i][agent_idx] for i in range(len(results))]
        agent_coordinate_list = coordinateList[agent_idx]
        connected_path = connect_points(agent_results, agent_coordinate_list, step=1)
        continuous_results.append(connected_path)

    for idx in range(len(continuous_results)):
        continuous_results[idx] = sorted(continuous_results[idx], key=lambda x: x[0])

    # np.save('final_paths.npy', continuous_results)
    # np.save('/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/PSRO_Path_Analysis/Lab_iter50_final_paths.npy', continuous_results)
    # np.save('/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/AgentNum_Analysis/2Agents_Lab_final_path.npy', continuous_results)
    np.save('/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/RewardCompare/lab/5Agents_Optimal50iter_final_path.npy', continuous_results)

    for agent_path in continuous_results:
        print(len(agent_path))
    
    return continuous_results

def plot_paths(paths):
    plt.figure(figsize=(10, 8))
    plt.grid(True)
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.title('Paths')
    plt.gca().invert_yaxis()
    plt.xticks(range(1, 24))
    plt.yticks(range(1, 22))
    plt.gca().set_aspect('equal', adjustable='box')
    
    for path in paths:
        x = [point[1] for point in path]
        y = [point[0] for point in path]
        plt.plot(x, y, marker='o')

    plt.legend(['Agent 1', 'Agent 2', 'Agent 3'])
    plt.show()

# plot_paths(continuous_results)
