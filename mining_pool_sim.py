import argparse
import numpy as np
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser('Blockchain Mining Reward Simulator')

parser.add_argument('--min_block_reward', type=int, default=100, help='Minimum Block reward')
parser.add_argument('--max_block_reward', type=int, default=110, help='Maximum Block reward - higher implies more transaction fees')

parser.add_argument('--alpha', type=float, default=2e-5, help='Probability of winning the mining race for the current actor')
parser.add_argument('--beta', type=float, default=1e-3, help='Probability of winning the mining race for the pool')
parser.add_argument('--gamma', type=float, default=3e-2, help='mining pool fee (percentage of earnings directed back to the pool)')
parser.add_argument('--include_transaction_rewards', action='store_true', help='Whether to account for transaction rewards in the simulation for "estimated" block reward schemes (pps, pplns)')
parser.add_argument('--timesteps', type=int, default=200000, help='number of timesteps to run')
parser.add_argument('--iters', type=int, default=30, help='number of iterations to run')

args = parser.parse_args()


all_rewards = np.zeros((4, args.iters))

for iter in range(args.iters):
    individual_rewards = []
    pplns_rewards = []
    proportional_rewards = []
    pps_rewards = []
    last_reward_t = 0
    shares_contributed = 0
    shares_total = 0
    
    for t in range(args.timesteps):
        shares_contributed += args.alpha
        shares_total += args.beta
        block_reward = np.random.randint(args.min_block_reward, args.max_block_reward)
        sample = np.random.rand()
        # conditionally get an individual mining reward
        if sample < args.alpha:
            individual_rewards.append(block_reward)
        else:
            individual_rewards.append(0)

        # conditionally get a PPLNS reward
        if sample < args.beta:
            estimated_br = (args.min_block_reward + args.max_block_reward) / 2. if args.include_transaction_rewards else args.min_block_reward
            pplns_reward = estimated_br * (args.alpha / args.beta) * (1 - args.gamma)
            pplns_rewards.append(pplns_reward)
            proportional_reward = block_reward * (shares_contributed / shares_total) * (1 - args.gamma)
            proportional_rewards.append(proportional_reward)
            last_reward_t = t
            shares_contributed = 0
            shares_total = 0
        else:
            pplns_rewards.append(0)
            proportional_rewards.append(0)
            

        # Always get a PPS reward
        estimated_br = (args.min_block_reward + args.max_block_reward) / 2. if args.include_transaction_rewards else args.min_block_reward
        pps_reward = estimated_br * (args.alpha / args.beta) * args.beta * (1 - args.gamma)
        pps_rewards.append(pps_reward)
        

    cum_individual_rewards = np.cumsum(individual_rewards)
    cum_pplns_rewards = np.cumsum(pplns_rewards)
    cum_pps_rewards = np.cumsum(pps_rewards)
    cum_proportional_rewards = np.cumsum(proportional_rewards)

    all_rewards[:, iter] = [cum_individual_rewards[-1], cum_pplns_rewards[-1], cum_pps_rewards[-1], cum_proportional_rewards[-1]]
    plt.figure()
    plt.suptitle('Mining Rewards')
    plt.title('alpha: {}, beta: {}, gamma: {}'.format(args.alpha, args.beta, args.gamma))
    plt.plot(cum_individual_rewards, label='Individual')
    plt.plot(cum_pplns_rewards, label='Pay-Per-Last-N-Shares')
    plt.plot(cum_pps_rewards, label='Pay-Per-Share')
    plt.plot(cum_proportional_rewards, label='Proportional')
    plt.ylabel('Cumulative Rewards')
    plt.xlabel('Timestep')
    plt.ylim(0, 600)
    plt.legend(loc='upper left')
    plt.savefig('images/mining_pool_{}.png'.format(iter))
    plt.close()

means = np.mean(all_rewards, axis=1)
stdevs = np.std(all_rewards, axis=1)

print('Individual: {} \pm {}'.format(means[0], stdevs[0]))
print('PPLNS: {} \pm {}'.format(means[1], stdevs[1]))
print('PPS: {} \pm {}'.format(means[2], stdevs[2]))
print('Proportional {} \pm {}'.format(means[3], stdevs[3]))

