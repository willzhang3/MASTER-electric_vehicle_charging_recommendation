import torch
import argparse
import logging
import numpy as np
import json
import time
import os
from env import Charging_Env
from multi_agent import Agent_MASTER

################################### Initialize Hyper-Parameters ###################################
parser = argparse.ArgumentParser(description='MASTER')
parser.add_argument('--gpu', type=str, default="0", help='Which GPU to use.')
parser.add_argument('--state', type=str, default="Def", help='The state of this running.')
parser.add_argument('--logmode', type=str, default="a", help='File mode of logging.')
parser.add_argument('--simulate', action="store_true", default=False, help='Debug.')
parser.add_argument('--test', action="store_true", default=False, help='Model Test.')
parser.add_argument('--encuda', action="store_false", default=True, help='Disable CUDA training.')
parser.add_argument('--noise', action="store_false", default=False, help='Add noise to action.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--n_pred', type=int, default=3, help='Max time_cost step the query not miss.')
parser.add_argument('--cp_time', type=int, default=30, help='Time duration to consider charging competition.')
parser.add_argument('--miss_time', type=int, default=46, help='request miss time.')
parser.add_argument('--interval', type=int, default=15, help='Time interval of one time step.')
parser.add_argument('--avg_charge_qt', type=float, default=48.96, help='Statistical Avg of charge.')
parser.add_argument('--std_charge_qt', type=float, default=10.43, help='Statistical Std of charge.')
parser.add_argument('--power_rate', type=float, default=0.5, help='power rate.')
parser.add_argument('--N', type=int, default=596, help='Number of charging station.')
parser.add_argument('--action_dim', type=int, default=50, help='Number of active agents.')
parser.add_argument('--n_spa', type=int, default=-1, help='Number of cs in sptial centralization.')
parser.add_argument('--hiddim', type=int, default=64, help='Dimension of critic.')
parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor in TD.')
parser.add_argument('--lr_c', type=float, default=5e-4, help='Initial learning rate.')
parser.add_argument('--lr_a', type=float, default=5e-4, help='Initial learning rate.')
parser.add_argument('--soft_tau_c', type=float, default=1e-3, help='Soft update ratio in critic params.')
parser.add_argument('--soft_tau_a', type=float, default=1e-3, help='Soft update ratio in actor params.')
parser.add_argument('--batch_size', type=int, default=32, help='Update batch_size.')
parser.add_argument('--freq_update', type=int, default=1, help='Hard update frequency.')
parser.add_argument('--replay_buffer_size', type=int, default=1000, help='Buffer capacity.')
parser.add_argument('--anneal_lr', action="store_true", default=False, help='anneal learning rate of actor and critic.')
parser.add_argument('--lr_decay', type=float, default=0.9, help='Lr dacay ratio.')
parser.add_argument('--dropout_rate', type=float, default=0, help='Dropout.')
parser.add_argument('--clip_norm', type=float, default=0.5, help='clip param.')
parser.add_argument('--ac_ratio', type=float, default=0.396, help='Statistical acceptance probability.')
parser.add_argument('--T_LEN', type=int, default=96, help='Number of time steps.')
parser.add_argument('--normalization', type=str, default="min-max", choices=['z-score', 'min-max'], help='File mode of logging.')
parser.add_argument('--load', action="store_true", default=False, help='load model.')
parser.add_argument('--load_path', type=str, default="def", help='load path.')
parser.add_argument('--temp', type=int, default=5, help='temperature in two critic.')

args = parser.parse_args()
args.n_spa = args.action_dim
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
logging.basicConfig(level = logging.INFO,filename='./logs/MASTER_{}_{}k_{}temp_{}gm_{}lr_a_{}lr_c.log'.format(args.state,args.action_dim,args.temp,args.gamma,args.lr_a,args.lr_c),filemode='{}'.format(args.logmode),\
                    format = '%(message)s')
args.state = "{}_top{}_temp{}".format(args.state,args.action_dim,args.temp)
logger = logging.getLogger(__name__)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.encuda and torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')
args.pid = os.getpid()
print(args)
logger.info(args)


################################### Load datas ###################################
# simulate data for code running
if(args.simulate):
    n_grids, N = 200, 100
    supply_dist = np.random.randint(0, 10, (45*args.T_LEN*args.interval, N, 2))
    # (T_LEN,n_grids,1) -- lambda
    demand_dist = np.random.randint(1, 10, (args.T_LEN, n_grids, 1)) 
    durations = np.random.randint(1, 3600, (n_grids, N))
    durations = np.clip(np.ceil(durations/60).astype(np.int32),0,args.miss_time)
    cs_surgrids = [np.random.choice(n_grids, size=4) for i in range(N)]
    # print(cs_surgrids)
    fee_24hour = np.random.uniform(1, 2, (N, 24))
    powers = np.random.randint(30, 200, (N,))
    # print(fee_24hour.max(), fee_24hour.min()) 

# load real data
else:
    PATH_DATA = "../exp_data/"
    # (N_DAY*T,N,2) -- supply at t
    supply_dist = np.load(os.path.join(PATH_DATA,"20190518-20190701_supply.npy")).transpose(1,0,2)
    # (T_LEN,n_grids,1) -- lambda
    demand_dist = None 
    # (n_grids,N) -- the eta (in second) from each grid to all cs
    durations = np.load(os.path.join(PATH_DATA,"durations.npy"))
    durations = np.clip(np.ceil(durations/60).astype(np.int32),0,args.miss_time)
    # (N, grid_ids), e.g., [[0,2],[1],[3,6,9],...] list of CS surrounding grid ids 
    with open(os.path.join(PATH_DATA,"cs_surgrids.list"),"r") as fp:
        cs_surgrids = np.asarray(json.load(fp))
    # (N, 24) 24 hours charging prices
    fee_24hour = np.load(os.path.join(PATH_DATA,"fees_24hour.npy"))
    # (N, ) charging powers of CS
    with open(os.path.join(PATH_DATA,"powers.list"),"r") as fp:
        powers = np.asarray(json.load(fp))

args.N = supply_dist.shape[1]
n_grids = durations.shape[0]

LOAD_PATH = "params/master_{}.pkl".format(args.load_path)
################################### Initialize env and agent ###################################
env = Charging_Env(args, n_grids, supply_dist, demand_dist, cs_surgrids, durations, fee_24hour, powers)
agent = Agent_MASTER(env, args, LOAD_PATH)
################################### Training ###################################
MAX_ITER = 60
N_DAY_TRAIN = 28
day_shuffle = []
for i in range(np.ceil(MAX_ITER/N_DAY_TRAIN).astype(np.int32)):
    days = list(range(N_DAY_TRAIN))
    np.random.shuffle(days)
    day_shuffle += days

max_reward = -1e8
for n_iter in range(MAX_ITER):
    st = time.time()
    RANDOM_SEED = n_iter + 33
    """ Env and agent reset
    """
    day = day_shuffle[n_iter]
    env.reset(RANDOM_SEED, day)  # generate all day supplies and demands
    agent.reset_agent()
    fee_costs,save_costs,time_costs, = [],[],[]
    losses_critic,losses_actor,rec_rewards = [],[],[]
    count_loss, count_query, count_rec, count_success_charge, count_success_charge_rec = 0, 0, 0, 0, 0
    for cur_t in range(0,args.T_LEN):
        fee_cost, save_cost, cost, loss_critic, loss_actor, n_query, n_rec, n_success_charge, n_success_charge_rec, rec_reward = agent.step(cur_t, n_iter)
        count_loss += len(loss_critic)
        count_query += n_query
        count_rec += n_rec
        count_success_charge += n_success_charge
        count_success_charge_rec += n_success_charge_rec
        fee_costs.extend(fee_cost)
        save_costs.extend(save_cost)
        time_costs.extend(cost)
        rec_rewards.extend(rec_reward)
        losses_critic.extend(loss_critic)
        losses_actor.extend(loss_actor)

    mean_fees = round(np.mean(fee_costs),3)
    sum_save_costs = round(np.sum(save_costs),2)
    mean_time_costs = round(np.mean(time_costs),2)

    mean_reward = round(np.mean(rec_rewards),3)
    mean_losses_critic = np.mean(losses_critic)
    mean_losses_actor = np.mean(losses_actor)

    state = {'actor':agent.Actor.state_dict(), 
            'critic_cwt':agent.Critic_cwt.state_dict(),
            'critic_fee':agent.Critic_fee.state_dict(),  
            'mean_fee':mean_fees,
            "save_cost":sum_save_costs,
            'time_cost':mean_time_costs,
            'success_rate':round(count_success_charge_rec/count_rec,3),
            'success_num':count_success_charge,
            'mean_reward': mean_reward,
            'n_iter':n_iter,
            }
    if(not os.path.exists("./params")):
        os.mkdir("./params")
    torch.save(state, 'params/master_{}_{}.pkl'.format(args.state,n_iter))

    print("n_iter: {}".format(n_iter))
    logging.info("n_iter: {}".format(n_iter))
    print("Date: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    logging.info("Date: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    print("Training - n_query, n_sc, n_rec, n_recsc, n_update: {},{},{},{},{}".format(count_query-1, count_success_charge, count_rec, count_success_charge_rec, count_loss))
    logging.info("Training - n_query, n_sc, n_rec, n_recsc, n_update: {},{},{},{},{}".format(count_query-1, count_success_charge, count_rec, count_success_charge_rec, count_loss))
    print("fee, save_cost, time_cost, success_rate, n_sc, reward: {}, {}, {}, {}, {}, {}".format\
        (mean_fees, sum_save_costs, mean_time_costs, round(count_success_charge_rec/count_rec,3),count_success_charge,mean_reward))
    logging.info("fee, save_cost, time_cost, success_rate, n_sc, reward: {}, {}, {}, {}, {}, {}".format\
        (mean_fees, sum_save_costs, mean_time_costs, round(count_success_charge_rec/count_rec,3),count_success_charge,mean_reward))
    print("loss_critic,loss_actor: {},{}".format(round(mean_losses_critic,3), round(mean_losses_actor,3)))
    logging.info("loss_critic,loss_actor: {},{}".format(round(mean_losses_critic,3), round(mean_losses_actor,3)))
    print("time_consuming: {}s".format(int(time.time()-st)))
    logging.info("time_consuming: {}s".format(int(time.time()-st)))
    
    ### evaluation ###
    """ Env and agent reset
    """
    st = time.time()
    count_query, count_rec, count_success_charge, count_success_charge_rec = 0, 0, 0, 0
    fee_costs,save_costs,time_costs,rec_rewards = [],[],[],[]
    days_test = [28,29,30]
    for d_test in days_test:
        env.reset(RANDOM_SEED, d_test) # generate all day supplies and demands
        agent.reset_agent()
        for cur_t in range(0,args.T_LEN):
            with torch.no_grad():
                fee_cost, save_cost, cost, _, _, n_query, n_rec, n_success_charge, n_success_charge_rec, rec_reward = agent.step(cur_t, n_iter, is_test=True)
            count_query += n_query
            count_rec += n_rec
            count_success_charge += n_success_charge
            count_success_charge_rec += n_success_charge_rec
            fee_costs.extend(fee_cost)
            save_costs.extend(save_cost)
            time_costs.extend(cost)
            rec_rewards.extend(rec_reward)

    mean_fees = round(np.mean(fee_costs),2)
    sum_save_costs = round(np.sum(save_costs),2)
    mean_time_costs = round(np.mean(time_costs),2)
    mean_reward = round(np.mean(rec_rewards),3)

    if(mean_reward>max_reward): 
        best_iter = n_iter
        best_fee = mean_fees
        best_savecost = sum_save_costs
        best_timecost = mean_time_costs
        best_scr = count_success_charge_rec/count_rec
        best_sc = count_success_charge
        max_reward = mean_reward

    print("Evaluation - n_query, n_sc, n_rec, n_recsc: {},{},{},{}".format(count_query-len(days_test), count_success_charge, count_rec, count_success_charge_rec))
    logging.info("Evaluation - n_query, n_sc, n_rec, n_recsc: {},{},{},{}".format(count_query-len(days_test), count_success_charge, count_rec, count_success_charge_rec))
    print("fee, save_cost, time_cost, success_rate, n_sc, reward: {}, {}, {}, {}, {}, {}".format\
        (mean_fees, sum_save_costs, mean_time_costs, round(count_success_charge_rec/count_rec,3),count_success_charge,mean_reward))
    logging.info("fee, save_cost, time_cost, success_rate, n_sc, reward: {}, {}, {}, {}, {}, {}".format\
        (mean_fees, sum_save_costs, mean_time_costs, round(count_success_charge_rec/count_rec,3),count_success_charge,mean_reward))
    print("best_iter, best_fee, best_savecost, best_time_cost, best_success_rate, best_success_count, max_reward: {}_iter, {}, {}, {}, {}, {}, {}".\
                format(best_iter, round(best_fee,3), round(best_savecost,3), round(best_timecost,3), round(best_scr,3), best_sc, max_reward))
    logging.info("best_iter, best_fee, best_savecost, best_time_cost, best_success_rate, best_success_count, max_reward: {}_iter, {}, {}, {}, {}, {}, {}".\
                format(best_iter, round(best_fee,3), round(best_savecost,3), round(best_timecost,3), round(best_scr,3), best_sc, max_reward))
    print("time_consuming: {}s".format(int(time.time()-st)))
    logging.info("time_consuming: {}s".format(int(time.time()-st)))
