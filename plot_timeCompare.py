import numpy as np
import matplotlib.pyplot as plt

def seconds_to_decimal_minutes(seconds):
    return seconds / 60

'----------Lab-----------'
stepLength = ['10', '20', '30', '40', '50']
CPPUTime = [113.3, 135.6, 150.2, 190.5, 230.6]
ILUCTime = [130.2, 153.6, 178.8, 204.2, 251.3]
A3CTime = [186.9, 225.0, 273.5, 300.1, 336.2]
DQNTime = [244.3, 288.7, 340.8, 384.4, 453.3]
GATime = [190.1, 243.2, 298.4, 347.2, 385.2]


'----------Meeting Room-----------'
# stepLength = ['5', '10', '15', '20', '25']
# CPPUTime = [61.2, 79.6, 93.4, 110.9, 136.4]
# ILUCTime = [86.4, 95.3, 123.4, 137.2, 145.9]
# A3CTime = [82.4, 97.6, 128.9, 138.1, 151.7]
# DQNTime = [91.1, 100.7, 122.8, 143.4, 152.3]
# GATime = [90.0, 106.2, 133.2, 153.6, 165.8]

CPPUTime = [seconds_to_decimal_minutes(time) for time in CPPUTime]
ILUCTime = [seconds_to_decimal_minutes(time) for time in ILUCTime]
A3CTime = [seconds_to_decimal_minutes(time) for time in A3CTime]
DQNTime = [seconds_to_decimal_minutes(time) for time in DQNTime]
GATime = [seconds_to_decimal_minutes(time) for time in GATime]

plt.plot(stepLength, CPPUTime, color='r', marker='o', label='CPPU')
plt.plot(stepLength, ILUCTime, color='b', marker='v', label='ILUC')
plt.plot(stepLength, A3CTime, color='g', marker='x', label='A3C-IPP')
plt.plot(stepLength, DQNTime, color='c', marker='p', label='DQN-IPP')
plt.plot(stepLength, GATime, color = 'orange', marker = '*', label = 'GA-IPP')

plt.xlabel('Iteration Numbers', size=18)
plt.ylabel('Training Time (min)', size=18)
plt.grid(color="grey", linestyle=':', linewidth=0.5)
plt.tick_params(labelsize=13)
plt.legend(loc='lower right', fontsize=14)
plt.yticks(np.arange(1, 2 + 1, 0.5))
plt.savefig('/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Fig/TimeCompare_lab.pdf', bbox_inches='tight')
plt.show()