from typing import List
from dataclasses import dataclass
import numpy as np
from multiRobotPathPlanner import MultiRobotPathPlanner

@dataclass
class DARPMap:
    rows: int = 10
    columns: int = 10

@dataclass
class DARPCoordinate:
    x: int = 0
    y: int = 0
    def to_index(self, map: DARPMap):
        return self.y * map.columns + self.x

@dataclass
class DARPProblem:
    map: DARPMap
    agents: List[DARPCoordinate]
    obstacles: List[DARPCoordinate]

    def solve(self, iterations: int = 1000):
        # Compute agent and obstacle indices...
        agent_indices = [agent.to_index(self.map) for agent in self.agents]
        obstacle_indices = [obstacle.to_index(self.map) for obstacle in self.obstacles]

        # Compute even portions...
        portions = np.ones(len(self.agents)) / len(self.agents)

        # Solve the problem...        
        solver = MultiRobotPathPlanner(
            nx = self.map.rows,
            ny = self.map.columns,
            notEqualPortions = True,
            initial_positions = agent_indices,
            portions = [0.3, 0.1, 0.2, 0.1, 0.3],  #    0.4, 0.2, 0.4
            obs_pos = obstacle_indices,
            visualization = True,
            MaxIter = iterations,
            CCvariation=0.01,
            randomLevel=0.0001, 
            dcells=2,
            importance=True
        )
        
        # self.solved,_ = solver.divideRegions()
        # self.solution = solver.BinaryRobotRegions
        return solver.best_case.paths

def fillObstacle(startRow, endROW, startColumn, endColumn):
    IterRow = np.arange(startRow, endROW+1)
    IterColumn = np.arange(startColumn, endColumn+1)
    coordinates = []
    for i in IterRow:
        for j in IterColumn:
            coordinates.append(DARPCoordinate(j, i))
    return coordinates

def parse_and_remove_duplicates(blocks):
    unique_coordinates = set()
    
    for block in blocks:
        start_x, start_y, end_x, end_y = [value // 2 for value in block]
        # print(f"Move from ({start_x}, {start_y}) to ({end_x}, {end_y})")
        unique_coordinates.add((start_x, start_y))
        unique_coordinates.add((end_x, end_y))

    sorted_coordinates = sorted(unique_coordinates, key=lambda coord: (coord[0], coord[1]))
    modified_coordinates = [(coord[0] + 1, coord[1] + 1) for coord in sorted_coordinates]

    return modified_coordinates

def find_PotentialPath():
    problem = DARPProblem(
        map = DARPMap(
            rows = 16,		# 行
            columns = 11,	# 列
        ),
        agents = [
            DARPCoordinate(0, 0), # agent1 upper right 10, 0
            DARPCoordinate(5, 0),  # agent2 center area	5, 7
            DARPCoordinate(10, 0), # agent3 lower left  0, 15
            DARPCoordinate(2, 15),
            DARPCoordinate(7, 15)
        ],
        obstacles = []	    #there is no blocks in the meeting room.
    )

    # begin to find the pathes in three areas
    # solution = problem.solve()
    # np.save('/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/AgentNum_Analysis/Meet_5Agents_Coverage_path.npy', solution)
    
    # np.save('/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/AgentNum_Analysis/Meet/ThreeAgent_(1,1)(8,1)(16,1)_path.npy', solution)
    # solution = np.load('/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/AgentNum_Analysis/Meet/ThreeAgent_(1,1)(8,1)(16,1)_path.npy', allow_pickle=True)
    solution = np.load('/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/AgentNum_Analysis/Meet_5Agents_Coverage_path.npy', allow_pickle=True)
    
    all_unique_coordinates = []
    for robot_solution in solution:
        blocks = robot_solution
        unique_coordinates = parse_and_remove_duplicates(blocks)
        all_unique_coordinates.append(unique_coordinates)
    return all_unique_coordinates