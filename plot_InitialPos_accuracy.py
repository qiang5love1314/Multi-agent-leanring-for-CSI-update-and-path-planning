import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

def main():
    stepLength = [ '5', '10', '15', '20', '25']

    '-----Lab-----'
    Lab_InitialPos_11121211 = [3.23, 3.17, 3.23, 3.38, 3.30]
    Lab_InitialPos_11121211_std = [1.70, 1.58, 1.84, 1.84, 1.80]
    Lab_InitialPos_11112123 = [3.20, 3.41, 3.34, 3.20, 3.21]
    Lab_InitialPos_11112123_std = [1.62, 1.88, 2.02, 1.52, 1.72]
    Lab_InitialPos_1231110211 = [2.98, 2.90, 3.11, 3.13, 3.34]
    Lab_InitialPos_1231110211_std = [1.55, 1.72, 1.63, 1.36, 1.58]

    plt.errorbar(stepLength, Lab_InitialPos_11121211, Lab_InitialPos_11121211_std, fmt='-o', ecolor='r', color='r', elinewidth=1, capsize=3, label='Initial Position=(1,1)(12,1)(21,1)')
    plt.errorbar(stepLength, Lab_InitialPos_11112123, Lab_InitialPos_11112123_std, fmt='-v', ecolor='b', color='b', elinewidth=1, capsize=3, label='Initial Position=(1,1)(1,12)(1,23)')
    plt.errorbar(stepLength, Lab_InitialPos_1231110211, Lab_InitialPos_1231110211_std, fmt='-x', ecolor='green', color='green', elinewidth=1, capsize=3, label='Initial Position=(1,23)(11,10)(21,1)')
    
    '-----Meet-----'
    # Meet_InitialPos_1181161 = [2.22 , 1.82, 1.85, 1.71, 1.77]
    # Meet_InitialPos_1181161_std = [1.14, 1.12, 1.33, 0.75, 1.22]
    # Meet_InitialPos_1111116 = [1.99 , 2.26, 1.93, 1.94, 2.07]
    # Meet_InitialPos_1111116_std = [0.89, 1.42, 0.93, 0.94, 1.35]
    # Meet_InitialPos_11186161 = [2.15, 2.1, 1.96, 1.84, 1.85]
    # Meet_InitialPos_11186161_std = [1.12, 1.21, 1.10, 1.19, 1.02]

    # plt.errorbar(stepLength, Meet_InitialPos_1181161, Meet_InitialPos_1181161_std, fmt='-o', ecolor='r', color='r', elinewidth=1, capsize=3, label='Initial Position=(1,1)(8,1)(16,1)')
    # plt.errorbar(stepLength, Meet_InitialPos_1111116, Meet_InitialPos_1111116_std, fmt='-v', ecolor='b', color='b', elinewidth=1, capsize=3, label='Initial Position=(1,1)(1,11)(1,6)')
    # plt.errorbar(stepLength, Meet_InitialPos_11186161, Meet_InitialPos_11186161_std, fmt='-x', ecolor='green', color='green', elinewidth=1, capsize=3, label='Initial Position=(1,11)(8,6)(16,1)')
    # fmt:'o' ',' '.' 'x' '+' 'v' '^' '<' '>' 's' 'd' 'p'

    plt.xlabel('Iteration Numbers',  size=20)
    plt.ylabel('Localization Error (m)', size=20)
    plt.grid(color="grey", linestyle=':', linewidth=0.5)
    plt.tick_params(labelsize=13)
    plt.legend(loc='lower right', fontsize=13)
    plt.ylim(0, 6)
    plt.tight_layout()
    # plt.savefig('/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Fig/InitialPos_LocAcc_Lab.pdf', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()