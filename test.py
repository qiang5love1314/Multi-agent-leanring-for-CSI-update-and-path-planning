# from MeetingRoom.MeetRewardCalculator import *
# import numpy as np
# from MeetingRoom.MeetMulAgtPath import *
# from MeetingRoom.Meet_PSRO  import *
# from MeetingRoom.Meet_continus_PSRO import findFinalPath, plot_paths

# firstly, we obtain the reward
# calculate = RewardCalculator()
# diagReward = calculate.calculateRewards()
# np.save('/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/MeetingRoom/Meet_rewards.npy', diagReward)

# Secondly, we obtain the coverage path by multi-agent
# find_PotentialPath()

# Thirdly, we obtain the optimal path by CPPU

# run_experiments(num_experiments=1, num_threads=20, iters=25, dim=args.dim, lr=LR, thresh=TH, psro=True)

# Finally, we obtain the final paths
# result = findFinalPath()
# plot_paths(result)

from Lab.Lab_PSRO import *
from Lab.LabMulAgtPath import *
from Lab.Lab_continus_PSRO import findFinalPath
# find_PotentialPath()
# run_experiments(num_experiments=1, num_threads=20, iters=50, dim=args.dim, lr=LR, thresh=TH, psro=True)
# findFinalPath()