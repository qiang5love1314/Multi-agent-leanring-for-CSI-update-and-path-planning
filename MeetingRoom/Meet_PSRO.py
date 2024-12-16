import numpy as np
import matplotlib.pyplot as plt
import time
import os
from scipy import stats
import pickle
import json
from numpy.random import RandomState
import argparse
import multiprocessing as mp
from MeetingRoom.MeetRewardCalculator import *
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)
np.random.seed(0)

parser = argparse.ArgumentParser(description='Policy space diversity for multi areas based-PSRO')
parser.add_argument('--dim', type=int, default=100)
parser.add_argument('--nb_iters', type=int, default=5)

args = parser.parse_args()

LR = 0.5
TH = 0.03

# 会议室场景智能体的起始坐标
# start_positions = np.array([
#     [10, 0],
#     [5, 7],
#     [0, 15]
# ])

# start_positions = np.array([
#     [0, 0],
#     [10, 15]
# ])

# start_positions = np.array([
#     [0, 0],
#     [10, 0],
#     [0, 15],
#     [10, 15]
# ])

start_positions = np.array([
    [0, 0],
    [5, 0],
    [10, 0],
    [2, 15],
    [7, 15]
])

reward_calculator = RewardCalculator()
coordinateList, sizes = reward_calculator.get_coordinatesList() # get the coverage path
rewards = reward_calculator.diagReward_selected()

time_string = time.strftime("%Y%m%d-%H%M%S")
PATH_RESULTS = os.path.join('results', time_string + '_' + str(args.dim))
if not os.path.exists(PATH_RESULTS):
    os.makedirs(PATH_RESULTS)

# Search over the pure strategies to find the BR to a strategy
def get_br_to_strat(strat, payoffs=None, verbose=False):
    row_weighted_payouts = strat @ payoffs
    br = np.zeros_like(row_weighted_payouts)
    br[np.argmin(row_weighted_payouts)] = 1
    if verbose:
        print(row_weighted_payouts[np.argmin(row_weighted_payouts)], "exploitability")
    return br

# Fictituous play as a nash equilibrium solver
def fictitious_play(iters=1000, payoffs=None, verbose=False):
    dim = payoffs.shape[0]
    pop = np.random.uniform(0, 1, (1, dim))
    pop = pop / pop.sum(axis=1)[:, None]
    averages = pop
    exps = []
    for i in range(iters):
        average = np.average(pop, axis=0)
        br = get_br_to_strat(average, payoffs=payoffs)
        exp1 = average @ payoffs @ br.T
        exp2 = br @ payoffs @ average.T
        exps.append(exp2 - exp1)
        # if verbose:
        #     print(exp, "exploitability")
        averages = np.vstack((averages, average))
        pop = np.vstack((pop, br))
    return averages, exps

# Solve exploitability of a nash equilibrium over a fixed population
def get_exploitability(pop, payoffs, iters=1000):
    emp_game_matrix = pop @ payoffs @ pop.T
    averages, _ = fictitious_play(payoffs=emp_game_matrix, iters=iters)
    strat = averages[-1] @ pop  # Aggregate
    test_br = get_br_to_strat(strat, payoffs=payoffs)
    exp1 = strat @ payoffs @ test_br.T
    exp2 = test_br @ payoffs @ strat
    return exp2 - exp1

def joint_loss(pop, payoffs, meta_nash, k, lambda_weight, lr):
    dim = payoffs.shape[0]

    br = np.zeros((dim,))
    values = []
    cards = []
    for i in range(dim):
        br_tmp = np.zeros((dim, ))
        br_tmp[i] = 1.

        aggregated_enemy = meta_nash @ pop[:k]
        value = br_tmp @ payoffs @ aggregated_enemy.T

        pop_k = lr * br_tmp + (1 - lr) * pop[k]
        pop_tmp = np.vstack((pop[:k], pop_k))
        M = pop_tmp @ payoffs @ pop_tmp.T
        metanash_tmp, _ = fictitious_play(payoffs=M, iters=1000)
        # L = np.diag(metanash_tmp[-1]) @ M @ M.T @ np.diag(metanash_tmp[-1])
        L = M @ M.T
        l_card = np.trace(np.eye(L.shape[0]) - np.linalg.inv(L + np.eye(L.shape[0])))

        cards.append(l_card)
        values.append(value)

    if np.random.randn() < lambda_weight:
        br[np.argmax(values)] = 1
    else:
        br[np.argmax(cards)] = 1

    return br

def psro_steps(iters=5, payoffs=None, verbose=False, seed=0,
                        num_learners=5, improvement_pct_threshold=.03, lr=.2, loss_func='dpp', full=False):
    dim = payoffs.shape[0]

    r = RandomState(seed)
    pop = r.uniform(0, 1, (1 + num_learners, dim))
    pop = pop / pop.sum(axis=1)[:, None]
    exp = get_exploitability(pop, payoffs, iters=100)
    exps = [exp]

    M = pop @ payoffs @ pop.T
    L = M @ M.T
    l_card = np.trace(np.eye(L.shape[0]) - np.linalg.inv(L + np.eye(L.shape[0])))
    l_cards = [l_card]

    learner_performances = [[.1] for i in range(num_learners + 1)]
    for i in range(iters):
        # Define the weighting towards diversity as a function of the fixed population size, this is currently a hyperparameter
        lambda_weight = 0.85
        if i % 5 == 0:
            print('iteration: ', i, ' exp full: ', exps[-1])
            print('size of pop: ', pop.shape[0])

        for j in range(num_learners):
            # first learner (when j=num_learners-1) plays against normal meta Nash
            # second learner plays against meta Nash with first learner included, etc.
            k = pop.shape[0] - j - 1
            emp_game_matrix = pop[:k] @ payoffs @ pop[:k].T
            meta_nash, _ = fictitious_play(payoffs=emp_game_matrix, iters=1000)
            population_strategy = meta_nash[-1] @ pop[:k]  # aggregated enemy according to nash

            if loss_func == 'br':
                # standard PSRO
                br = get_br_to_strat(population_strategy, payoffs=payoffs)
            else:
                # Diverse PSRO
                br = joint_loss(pop, payoffs, meta_nash[-1], k, lambda_weight, lr)
                br_orig = get_br_to_strat(population_strategy, payoffs=payoffs)

            # Update the mixed strategy towards the pure strategy which is returned as the best response to the
            # nash equilibrium that is being trained against.
            pop[k] = lr * br + (1 - lr) * pop[k]
            performance = pop[k] @ payoffs @ population_strategy.T + 1  # make it positive for pct calculation
            learner_performances[k].append(performance)

            # if the first learner plateaus, add a new policy to the population
            if j == num_learners - 1 and performance / learner_performances[k][-2] - 1 < improvement_pct_threshold:
                learner = np.random.uniform(0, 1, (1, dim))
                learner = learner / learner.sum(axis=1)[:, None]
                pop = np.vstack((pop, learner))
                learner_performances.append([0.1])

        # calculate exploitability for meta Nash of whole population
        exp = get_exploitability(pop, payoffs, iters=1000)
        exps.append(exp)

        M = pop @ payoffs @ pop.T
        L = M @ M.T
        l_card = np.trace(np.eye(L.shape[0]) - np.linalg.inv(L + np.eye(L.shape[0])))
        l_cards.append(l_card)

    return pop, exps, l_cards

def calculate_payoffs(rewards):
    num_agents = 5  # len(rewards)
    max_dim = max(len(agent_rewards) for agent_rewards in rewards)
    payoffs = np.zeros((max_dim, max_dim))

    for i in range(num_agents):
        agent_rewards = rewards[i]
        dim = len(agent_rewards)
        random_noise = np.random.normal(0.5, 0.2, dim)
        normalized_rewards = (agent_rewards - np.min(agent_rewards)) / (np.max(agent_rewards) - np.min(agent_rewards))
        agent_payoffs = normalized_rewards + random_noise
        payoffs[:dim, :dim] += agent_payoffs

    return payoffs

def psro_steps_fixed_starting_positions(start_positions, movable_coordinates, iters=10, payoffs=None, verbose=False, seed=0,
                        num_learners=5, improvement_pct_threshold=.03, lr=.2, loss_func='dpp', full=False):
    action_space_sizes = sizes
    np.random.seed(seed)
    payoffs = calculate_payoffs(payoffs)
    dim = payoffs.shape[0]
    
    r = RandomState(seed)
    pop = r.uniform(0, 1, (1 + num_learners, dim))
    pop = pop / pop.sum(axis=1)[:, None]
    exp = get_exploitability(pop, payoffs, iters=100)
    exps = [exp]
    
    M = pop @ payoffs @ pop.T
    L = M @ M.T
    l_card = np.trace(np.eye(L.shape[0]) - np.linalg.inv(L + np.eye(L.shape[0])))
    l_cards = [l_card]

    learner_performances = [[.1] for i in range(num_learners + 1)]
    
    best_ranges = []  # 存储最佳移动路径
    
    for i in range(iters):
        lambda_weight = 0.85
        if i % 5 == 0:
            print('iteration: ', i, ' exp full: ', exps[-1])

        for j in range(num_learners):
            k = pop.shape[0] - j - 1
            emp_game_matrix = pop[:k] @ payoffs @ pop[:k].T
            meta_nash, _ = fictitious_play(payoffs=emp_game_matrix, iters=1000)
            population_strategy = meta_nash[-1] @ pop[:k]

            if loss_func == 'br':
                br = get_br_to_strat(population_strategy, payoffs=payoffs)
            else:
                br = joint_loss(pop, payoffs, meta_nash[-1], k, lambda_weight, lr)
                br_orig = get_br_to_strat(population_strategy, payoffs=payoffs)

            dim = action_space_sizes[j]
            pure_strategies = np.zeros((len(movable_coordinates[j]), dim))
            
            br_subset = br[:dim]
            weighted_strategies = np.dot(pure_strategies, br_subset)
            pop[j][:dim] = lr * weighted_strategies + (1 - lr) * pop[j][:dim]
            performance = pop[j] @ payoffs @ population_strategy.T + 1
            
            learner_performances[j].append(performance)

            if j == num_learners - 1 and performance / learner_performances[j][-2] - 1 < improvement_pct_threshold:
                learner = np.random.uniform(0, 1, pop[0].shape)
                # learner[start_positions[j]] = 1.0  # 将该智能体的起始位置的概率设置为1.0
                learner = learner / learner.sum()
                pop = np.vstack((pop, learner))
                learner_performances.append([0.1])

        # 提取最佳移动路径
        best_path = []
        for j in range(num_learners):
            dim = action_space_sizes[j]
            best_strategy = pop[-(j+1)][:dim] / pop[-(j+1)][:dim].sum()  # 从后往前取
            best_move_index = np.argmax(best_strategy)
            best_path.append(movable_coordinates[j][best_move_index])  # best_move_index的tolist()方法是后加的，agent=2之前没有
        best_ranges.append(best_path)
        print(best_ranges)

        exp = get_exploitability(pop, payoffs, iters=100)
        exps.append(exp)

        M = pop @ payoffs @ pop.T
        L = M @ M.T
        l_card = np.trace(np.eye(L.shape[0]) - np.linalg.inv(L + np.eye(L.shape[0])))
        l_cards.append(l_card)
    
    # np.save('/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/MeetingRoom/Meet_optimal_path_CPPU.npy', best_ranges)
    # np.save('/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/PSRO_Path_Analysis/Meet_iter25_PSRO_Path.npy', best_ranges)
    # np.save('/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/AgentNum_Analysis/Meet/Optimal25iter_ThreeAgent_(1,1)(8,1)(16,1)_path.npy',best_ranges)
    np.save('/Users/zhuxiaoqiang/Desktop/IEEE Trans/BJTU-third multi agent/main/Experiments/RewardCompare/meet/Meet_5Agents_Optimal25iter_path.npy', best_ranges)
    return pop, exps, l_cards

def run_experiment(param_seed):
    params, seed = param_seed
    iters = params['iters']
    num_threads = params['num_threads']
    dim = params['dim']
    lr = params['lr']
    thresh = params['thresh']
    psro = params['psro']
    
    psro_exps = []
    psro_cardinality = []

    print('Experiment: ', seed + 1)
    np.random.seed(seed)
    W = np.random.randn(dim, dim)
    S = np.random.randn(dim, 1)
    payoffs = 0.5 * (W - W.T) + S - S.T
    payoffs /= np.abs(payoffs).max() 

    if psro:
        # print('PSRO')
        # pop, exps, cards = psro_steps(iters=iters, num_learners=1, seed=seed+1,
        #                               improvement_pct_threshold=thresh, lr=lr,
        #                               payoffs=payoffs, loss_func='br')
        pop, exps, cards = psro_steps_fixed_starting_positions(iters=iters, start_positions=start_positions, 
                                                               movable_coordinates=coordinateList, payoffs=rewards, num_learners=5)
        psro_exps = exps
        psro_cardinality = cards

    return {
        'psro_exps': psro_exps,
        'psro_cardinality': psro_cardinality,
    }

def run_experiments(num_experiments=1, iters=40, num_threads=20, dim=60, lr=0.6, thresh=0.001, logscale=True, psro=False):
    params = {
        'num_experiments': num_experiments,
        'iters': iters,
        'num_threads': num_threads,
        'dim': dim,
        'lr': lr,
        'thresh': thresh,
        'psro': psro,
    }

    psro_exps = []
    psro_cardinality = []

    with open(os.path.join(PATH_RESULTS, 'params.json'), 'w', encoding='utf-8') as json_file:
        json.dump(params, json_file, indent=4)

    pool = mp.Pool()
    result = pool.map(run_experiment, [(params, i) for i in range(num_experiments)])
    

    for r in result:
        psro_exps.append(r['psro_exps'])
        psro_cardinality.append(r['psro_cardinality'])

    d = {
        'psro_exps': psro_exps,
        'psro_cardinality': psro_cardinality,
    }
    pickle.dump(d, open(os.path.join(PATH_RESULTS, 'data.p'), 'wb'))

    def plot_error(data, label=''):
        data_mean = np.mean(np.array(data), axis=0)
        error_bars = stats.sem(np.array(data))
        # print(error_bars)
        plt.plot(data_mean, label=label)
        plt.fill_between([i for i in range(data_mean.size)],
                         np.squeeze(data_mean - error_bars),
                         np.squeeze(data_mean + error_bars), alpha=alpha)

    alpha = .4
    for j in range(2):
        fig_handle = plt.figure()

        if psro:
            if j == 0:
                plot_error(psro_exps, label='PSRO')
            elif j == 1:
                plot_error(psro_cardinality, label='PSRO')
        
        plt.legend(loc="upper right")
        plt.title('Dim {:d}'.format(args.dim))

        if logscale and (j==0):
            plt.yscale('log')
        plt.show()
        # plt.savefig(os.path.join(PATH_RESULTS, 'figure_'+ str(j) + '.pdf'))

# if __name__ == "__main__":
#     run_experiments(num_experiments=1, num_threads=20, iters=args.nb_iters, dim=args.dim, lr=LR, thresh=TH, psro=True)
    # psro_steps_fixed_starting_positions(start_positions=start_positions, movable_coordinates=coordinateList, payoffs=None, num_learners=3)