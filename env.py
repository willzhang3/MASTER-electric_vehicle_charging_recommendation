import numpy as np
import torch

TorchFloat = None
TorchLong = None

class RunningStat(object):

    def __init__(self, shape):
        self._n = np.int64(0)
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)
    def push(self, x):
        x = np.asarray(x)
        n_sample, n_feat = x.shape
        assert n_feat == self._M.shape[-1]
        self._n += n_sample
        oldM = self._M.copy()
        self._M[...] = oldM + (x - oldM).sum(axis=0)/self._n # mean
        self._S[...] = self._S + ((x - oldM)*(x - self._M)).sum(axis=0)/(self._n-1) # sum of (x-x^bar)^2
    @property
    def n(self):
        return self._n
    @property
    def mean(self):
        return self._M
    @property
    def var(self):
        return self._S
    @property
    def std(self):
        return np.sqrt(self.var)
    @property
    def shape(self):
        return self._M.shape
    def reset(self):
        pass
        # self._n = 0
        # self._M = np.zeros(self._shape)
        # self._S = np.zeros(self._shape)

class Identity:
    '''
    A convenience class which simply implements __call__
    as the identity function
    '''
    def __call__(self, x, *args, **kwargs):
        return x

    def reset(self):
        pass

class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """
    def __init__(self, shape, center=True, scale=True, clip=None):
        assert shape is not None
        self.center = center
        self.scale = scale
        self.clip = clip
        self.rs = RunningStat(shape)
        # self.prev_filter = prev_filter

    def __call__(self, x, **kwargs):
        # x = self.prev_filter(x, **kwargs)
        self.rs.push(x)
        # print(self.rs.mean,self.rs.std)
        if self.center:
            x = x - self.rs.mean
        if self.scale:
            if self.center:
                x = x / (self.rs.std + 1e-8)
            else:
                diff = x - self.rs.mean
                diff = diff/(self.rs.std + 1e-8)
                x = diff + self.rs.mean
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def reset(self):
        # self.prev_filter.reset()
        self.rs.reset()

class RewardFilter:
    """
    1. update return
    2. divide reward by std(return) *without* subtracting and adding back mean
    """
    def __init__(self, shape, gamma, clip=None):
        assert shape is not None
        self.gamma = gamma
        # self.prev_filter = prev_filter
        self.rs = RunningStat(shape)
        self.ret = np.zeros(shape)
        self.clip = clip

    def __call__(self, x, **kwargs):
        # x = self.prev_filter(x, **kwargs)
        self.ret = self.ret * self.gamma + x
        self.rs.push(self.ret)
        x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x
    
    def reset(self):
        self.ret = np.zeros_like(self.ret)
        # self.prev_filter.reset()
        self.rs.reset()

class Charging_Env:
    def __init__(self, args, n_grids, supply_dist=None, demand_dist=None, cs_surgrids=None, \
            durations=None, fee_24hour=None, powers=None):
        global TorchFloat, TorchLong
        TorchFloat = torch.cuda.FloatTensor if args.device == torch.device('cuda') else torch.FloatTensor
        TorchLong = torch.cuda.LongTensor if args.device == torch.device('cuda') else torch.LongTensor

        # env params
        self.grid2cs_duration = durations 
        self.cs_surgrids = cs_surgrids
        self.powers = powers
        self.charge_fee24 = fee_24hour
        self.N, self.n_grids = args.N, n_grids
        self.interval = args.interval # 15min
        self.avg_charge_power = args.avg_charge_power
        self.std_charge_power = args.std_charge_power
        self.power_rate = args.power_rate
        self.T = args.T_LEN*args.interval
        self.T_LEN = args.T_LEN
        self.n_pred = args.n_pred
        self.limit_waiting_time = self.n_pred*self.interval
        self.miss_time = args.miss_time
        self.cp_time = args.cp_time
        self.gamma = args.gamma
        self.action_dim = args.action_dim
        self.normalization = args.normalization
        self.query_idx = 0
        self.sup_i = 1

        # state features
        self.powers_rec = powers.reshape(1,self.N,1).repeat(self.T_LEN*self.interval,axis=0)
        self.cs_idx = np.arange(self.N).reshape(1,self.N,1).repeat(self.T_LEN*self.interval,axis=0) # (T,N,1)
        self.supply = np.zeros((self.T_LEN,self.N,1),dtype=np.int16) # supply at t
        self.supply_dist = supply_dist # (T_LEN,N,2) -- mu,sigma
        self.demand = np.zeros((self.T_LEN,n_grids,1),dtype=np.int16) # demand about future 15min
        self.demand_dist = demand_dist # (T_LEN,n_grids,1) -- lambda
        self.cs_demand = np.zeros((self.T_LEN*self.interval,self.N,1),dtype=np.int16) # demand in neighbor grids of cs
        self.t_step = np.arange(self.T_LEN,dtype=np.int16).reshape(self.T_LEN,1,1).repeat(self.N,axis=1) 
        self.t_step = self.t_step.repeat(self.interval,axis=0) # (T,N,1)
        self.chargefee = np.expand_dims(fee_24hour.transpose(1,0),axis=-1).repeat(4,axis=0)[:self.T_LEN].repeat(self.interval,axis=0)
        self.querys = None
        self.base_state = None
        
        # state and event
        self._state = None
        self._event_in = None
        
        # reset env
        self.RANDOM_SEED = 33
        
        ## normalization
        clip_obs = clip_rew = 10
        # state normalization
        self.state_filter = ZFilter(shape=[1,5], center=True, clip=clip_obs)
        # rewards normalization
        self.reward_filter = ZFilter(shape=(), center=True, clip=clip_rew)

    def grid2allcs_durations(self, grid_idx, expand_dim=True):
        cs_duration = self.grid2cs_duration[grid_idx] # (N,)
        cs_duration = np.expand_dims(cs_duration,axis=-1) if expand_dim else cs_duration
        return cs_duration
    
    def action_sample(self, ind):
        k = len(ind)
        idx = np.random.randint(k)
        return ind[idx]
    
    def est_power(self, cs_idx):
        charge_power = np.random.normal(self.avg_charge_power,self.std_charge_power)
        return charge_power

    def est_time(self, cs_idx):
        charge_power = self.est_power(cs_idx)
        charge_time = (charge_power / (self.powers[cs_idx]*self.power_rate)) * 60
        return charge_time, charge_power

    def get_cp(self, cs_idx, hour):
        return self.charge_fee24[cs_idx,hour]
    
    def supply_load(self, day):
        """ Load real-world data
        """
        self.supply = self.supply_dist[day*self.T:day*self.T+self.T,...]
        self.supply = np.where(self.supply>=0, self.supply, 0)
        return self.supply
    
    def supply_generation(self):
        """ Gaussian distribution
        """
        for cur_t in range(self.T):
            mu, sigma = self.supply_dist[cur_t,:,0],self.supply_dist[cur_t,:,1] # (T,N,1)
            self.supply[cur_t] = np.expand_dims(np.random.normal(mu,sigma),axis=-1).astype(np.int16)
        self.supply = np.where(self.supply>=0, self.supply, 0)
        return self.supply 
    
    def demand_load(self, day):
        """ Load real-world querys
        """
        dates = ['20190518', '20190519', '20190520', '20190521', '20190522', '20190523', '20190524', '20190525', '20190526', '20190527', '20190528', '20190529', '20190530', '20190531',\
         '20190601', '20190602', '20190603', '20190604', '20190605', '20190606', '20190607', '20190608', '20190609', '20190610', '20190611', '20190612', '20190613', '20190614',\
         '20190615', '20190616', '20190617', '20190618', '20190619', '20190620', '20190621', '20190622', '20190623', '20190624', '20190625', '20190626', '20190627', '20190628',\
         '20190629', '20190630', '20190701']
        self.demand = self.demand.repeat(self.interval,axis=0) # demand about future 15min
        self.querys = [[] for t in range(self.T_LEN*self.interval)] # (T,[(grid_idx, query_time, query_idx)...])
        self.query_idx = 0
        with open("../exp_data/request/{}".format(dates[day]),"r") as fp:
            for line in fp:
                col = list(map(int,line.strip().split("\t")))
                grid_idx, t_min, target_cs= col[0], col[1], col[-1]
                query_tp = (grid_idx,t_min,self.query_idx,target_cs)
                if(t_min >= self.T_LEN*self.interval):
                    continue
                self.querys[col[1]].append(query_tp)
                self.query_idx += 1
                for delta in range(15):
                    tt = t_min - delta - 1
                    if(tt >= 0): self.demand[tt,grid_idx] += 1
        return self.demand, self.querys

    def demand_generation(self):
        """ Poisson distribution
        """ 
        self.querys = [[] for t in range(self.T_LEN*self.interval)] # (T,[(grid_idx, query_time)...])
        self.query_idx = 0
        for cur_t in range(self.T_LEN):
            lam = self.demand_dist[cur_t,:,0] # (n_grids,1) -- lambda(mean)
            self.demand[cur_t] = np.expand_dims(np.random.poisson(lam),axis=-1) # (n_grids,1) -- number of querys in each grid
            # time of querys out of uniform distribution
            # (n_grid, n_querys)
            grid_querys = [np.random.choice(range(self.interval),*num_query_grid) for num_query_grid in self.demand[cur_t]]
            t_querys = []
            for grid_idx, query_time_offset in enumerate(grid_querys): # one grid
                tmp = list(zip([grid_idx]*len(query_time_offset),cur_t*self.interval+query_time_offset))
                t_querys.extend(tmp)
            t_querys = sorted(t_querys,key=lambda x:x[1]) # sorted by query_time
            for query_tp in t_querys:
                query_tp = (query_tp[0],query_tp[1],self.query_idx)
                self.querys[query_tp[1]].append(query_tp)
                self.query_idx += 1
        self.demand = self.demand.repeat(self.interval,axis=0)
        return self.demand, self.querys
    
    def cs_neighbor_demand(self):
        for cs_idx in range(self.N):
            surgrid = self.cs_surgrids[cs_idx]
            for grid_idx in surgrid:
                self.cs_demand[:,cs_idx] += self.demand[:,grid_idx]
        return self.cs_demand
        
    def get_state(self, t):
        return self._state[t]

    def get_observe(self, n_q, state, duration, inds=None, top_k=None):
        if(top_k is None): top_k = self.action_dim
        if(inds is None):
            inds = np.argpartition(duration.squeeze(axis=-1),top_k-1)[...,:top_k] # (n_q, k) or (k)
            inds = np.expand_dims(inds,axis=-1)
        observe = np.take_along_axis(state,inds,axis=-2)
        return observe, inds

    def get_observe_torch(self, state, duration, inds=None, top_k=None):
        if(top_k is None): top_k = self.action_dim
        if(inds is None):
            inds = np.argpartition(duration.squeeze(axis=-1), top_k-1)[...,:top_k] # (n_q, k)
            inds = torch.from_numpy(inds).type(TorchLong).unsqueeze(dim=-1)
        observe = state.gather(-2, inds.repeat(1,1,state.shape[-1]))
        return observe, inds

    def get_query(self, t):
        if(t<self.T):
            return self.querys[t]
        elif(t==self.T+self.limit_waiting_time-1): # a virtual query
            return [(0,self.T+self.limit_waiting_time-1,-1)] 

    def get_integrated_obs(self, st, et=None):
        if(et == None):
            return self._integrated_obs[st]
        return self._integrated_obs[st:et] # [m, int_obs(1,F)]
    
    def reset_randomseed(self, random_seed):
        self.RANDOM_SEED = random_seed
    
    def reset_state(self, state):
        self._state = state
        self._integrated_state = [None for _ in range(self.query_idx)] # each query correspond to a state
        self._centralized_state = [None for _ in range(self.query_idx)] # each query correspond to a centralized state 
        self._last_queryidx = [[] for _ in range(self.query_idx)]
        self._joint_action = [None for _ in range(self.query_idx)]
        self._dones = [None for _ in range(self.query_idx)]
        self._etc = [None for _ in range(self.query_idx)]
        self._rewards = [None for _ in range(self.query_idx)]
        self._query_info = [None for _ in range(self.query_idx)]

    def reset_event(self):
        self._event_arrival = [[] for _ in range(self.T+self.limit_waiting_time+1)] # (T,)
        self._event_in = [[] for _ in range(self.T+self.limit_waiting_time+1)] # (T,)
        self._event_update = [[] for _ in range(self.T+self.limit_waiting_time+1)] # (T,)
    
    def init_state(self):
        # state features
        self.supply = np.zeros((self.T_LEN,self.N,1),dtype=np.int16) # supply at t
        self.demand = np.zeros((self.T_LEN,self.n_grids,1),dtype=np.int16) # demand about future 15min
        self.cs_demand = np.zeros((self.T_LEN*self.interval,self.N,1),dtype=np.int16) # demand in neighbor grids of cs
        self.querys = None
        self.base_state = None

    def reset(self, RANDOM_SEED, day):
        self.reset_randomseed(RANDOM_SEED)
        np.random.seed(self.RANDOM_SEED)
        self.state_filter.reset() # state normalization
        self.reward_filter.reset() # reward normalization
        self.init_state()
        self.supply = self.supply_load(day)
        self.demand, self.querys = self.demand_load(day)
        self.cs_demand = np.clip(self.cs_neighbor_demand(),0,20) # (T,N,1)
        # cs_idx, supply, demand, t_step
        self.base_state = np.concatenate([self.powers_rec,self.supply,self.cs_demand,self.t_step,self.chargefee,self.cs_idx],axis=-1) #(T,N,F)
        # reset state and event_in 
        self.reset_state(self.base_state)
        self.reset_event()

    def arrival_step(self, t):
        """ event1: dispose vehicle arrival at t
        """
        sup_i = self.sup_i # supply index in state
        t_event_arrival = self._event_arrival[t] 
        success_cnt_rec, success_cnt = 0, 0 
        time_costs = []
        fee_costs = []
        save_costs = []
        rec_rewards = []
        for visit_tp in t_event_arrival:  # arrive at t
            st, query_grid, query_idx, action_duration, cs_idx, action, joint_action, is_rec, o_cs_idx= visit_tp
            time_cost = max(t - st, action_duration)
            time_cost = time_cost if(time_cost<=self.limit_waiting_time) else self.miss_time 
            if(t<self.T):  # tn is the time vehicle in (or until exceed limit_waiting_time)
                done = 0; tn = t
            else: 
                done = 1; tn = t%self.T
            success_charge = False
            charge_time = 0
            # simulate when vehicle in or miss
            while(time_cost<=self.limit_waiting_time):
                if(self._state[tn,cs_idx,sup_i]>=1): # vehicle success charge
                    in_t = tn
                    charge_time, charge_power = np.ceil(self.est_time(cs_idx)).astype(np.int32) # gaussian distribution
                    leave_t = tn + charge_time # leave time 
                    self._state[in_t:leave_t,cs_idx,sup_i] -= 1
                    if(leave_t > self.T): 
                        self._state[:leave_t%self.T,cs_idx,sup_i] -= 1
                    success_charge = True
                    # cs responses this query transfer to state that vehicle in
                    break
                else:  # wait this minute
                    self._state[tn,cs_idx,sup_i] -= 1
                    tn += 1
                    if(tn>=self.T):
                        tn = tn % self.T
                        done = 1
                    time_cost += 1 
            if(success_charge): 
                success_cnt += 1
                if(done == 1): tn += self.T
                # 0.64, 1.73, 2.3
                fee = self.get_cp(cs_idx, hour=int(tn%self.T/60))
                o_fee = self.get_cp(o_cs_idx, hour=int(tn%self.T/60))
                if(is_rec): # successful charging and accept recommendation
                    success_cnt_rec += 1
                    fee_costs.append(fee)
                    save_costs.append((o_fee-fee)*charge_power)

                reward_cwt = (-time_cost+60)/60
                reward_fee = (-fee+2.8)/2
                reward_add = -time_cost + -fee*30
            else: 
                leave_t = st + time_cost
                time_cost = self.miss_time
                tn = st+self.limit_waiting_time+1
                reward_cwt = 0
                reward_fee = 0
                reward_add = -60 + -2.8*30

            # reward normalization
            # if(self.normalization == "z-score"): 
            #     reward = self.reward_filter(reward)
            # elif(self.normalization == "min-max"): 
            #     reward = reward

            # for model update
            if(is_rec):
                time_costs.append(time_cost)
                rec_rewards.append(reward_add)
                ### for model update
                self._event_in[tn].append((st, query_grid, query_idx, action_duration, t, leave_t, cs_idx, action, joint_action, reward_cwt, reward_fee, done))

        self._event_arrival[t] = [] # discard arrival event at t
        return fee_costs, save_costs, time_costs, success_cnt, success_cnt_rec, rec_rewards
            
    def query_step(self, n_q, t_querys, action_cs_idx, action, joint_action, is_rec, original_cs_idx):
        """ event2: dispose charging query at t
        """
        for i in range(n_q):
            cs_idx = action_cs_idx[i]
            if(cs_idx == -1): continue
            query_tp = t_querys[i]
            st, query_grid, query_idx = query_tp[1], query_tp[0], query_tp[2]
            o_cs_idx = original_cs_idx[i]
            cs_durations = self.grid2allcs_durations(query_grid, expand_dim=False) # (N,)  [1,60]
            action_duration = cs_durations[cs_idx] 
            cs_eta = st + action_duration
            if(cs_eta > self.T+self.limit_waiting_time -1): cs_eta = self.T+self.limit_waiting_time -1 # end of day
            visit_tp = (st, query_grid, query_idx, action_duration, cs_idx, action[i:i+1], joint_action[i:i+1], is_rec[i], o_cs_idx)
            self._event_arrival[cs_eta].append(visit_tp)
            
    def index_step(self, t, n_q, t_querys, LASTUPDATE_T):
        """ derive transition of which next_state is at t
        """
        # update after 30 minute
        t_update = (t+self.cp_time) % (self.T+self.limit_waiting_time+1)
        for i in range(n_q): # there is n_q next_state 
            query_tp = t_querys[i]
            next_query_grid, next_st, next_query_idx = query_tp[:3]
            self._query_info[next_query_idx] = [next_query_grid, next_st]
            for in_t in range(LASTUPDATE_T+1,t+1): # update arrival in this time range
                t_event_in = self._event_in[in_t]
                for visit_tp in t_event_in:  # arrive at t
                    st, query_grid, query_idx, action_duration, in_t, leave_t, cs_idx, action, joint_action, reward_cwt, reward_fee, done = visit_tp
                    self._last_queryidx[next_query_idx].append(query_idx)
                    self._event_update[t_update].append(next_query_idx)
                    if(i>0): continue   

                    acc_reward_cwt = .0 # discount accumulative reward from (st+1) to t+1
                    acc_reward_fee = .0
                    for tt in range(st+1,t+1):
                        tt_event_in = self._event_in[tt]
                        discount = self.gamma**(tt-st-1)
                        for ele in tt_event_in:
                            acc_reward_cwt += ele[-3]*discount
                            acc_reward_fee += ele[-2]*discount
                    self._rewards[query_idx] = [acc_reward_cwt, acc_reward_fee]
                    self._joint_action[query_idx] = joint_action
                    self._etc[query_idx] = [t-st,query_grid,cs_idx,st,in_t,leave_t] # dt, rew
                    self._dones[query_idx] = done

    def get_centralized_state(self, state, future_supply):
        """ Args:
                state: (1,N,F)
                future_supply: (cp_time, N)
        """
        future_sup = future_supply.transpose(1,0)[np.newaxis,...]
        future_sup = np.where(future_sup<=10,future_sup,10)/10
        future_sup = torch.from_numpy(future_sup).type(TorchFloat)
        cent_state = torch.cat([state, future_sup],dim=-1) # (1,N,F+30)
        return cent_state

    def transition_step(self, t):
        """ event3: derive transition of which next_state is at t
        """
        ### transition
        states = []
        cent_states = []
        joint_actions = []
        next_states = []
        next_cent_states = []
        durations = []
        next_durations = []
        rewards = []
        dones = []
        etc = []
        t_update_queryidxs = self._event_update[t]
        for query_idx in t_update_queryidxs: # there is n_q next_state (or sample one)
            query_grid, st = self._query_info[query_idx]
            et = t + 1
            next_state = self._integrated_state[query_idx] # (1,N,F)
            next_cs_durations = self.grid2allcs_durations(query_grid)
            for last_query_idx in self._last_queryidx[query_idx]:
                _, last_query_grid, last_cs_idx, last_st, last_in_t, last_leave_t = self._etc[last_query_idx]
                future_supply = self._state[last_st+1:et,:,self.sup_i].copy()
                last_in_t_offset = last_in_t - last_st - 1 
                last_leave_t_offset = last_leave_t - last_st - 1
                st_offset = st - last_st - 1
                if(et>self.T):
                    concat_supply = self._state[:(st+self.cp_time)%self.T,:,self.sup_i].copy()
                    future_supply = np.concatenate([future_supply,concat_supply],axis=0)
                future_supply[last_in_t_offset:last_leave_t_offset,last_cs_idx] += 1
                # print(future_supply.shape,st,query_idx,last_st,last_query_idx)

                ### last state 
                state = self._integrated_state[last_query_idx]
                states.append(state)
                cent_state = self.get_centralized_state(state,future_supply[:self.cp_time]) # add future supply
                cent_states.append(cent_state)

                cs_durations = self.grid2allcs_durations(last_query_grid)
                durations.append(cs_durations)

                joint_action = self._joint_action[last_query_idx]
                joint_actions.append(joint_action)
                
                reward = self._rewards[last_query_idx] # [acc_reward_cwt, acc_reward_fee]
                # if(np.random.random() > 0.999): print(reward)

                rewards.append(reward)
                dones.append(self._dones[last_query_idx])
                etc.append(self._etc[last_query_idx])
                
                ### state
                next_states.append(next_state)
                next_cent_state = self.get_centralized_state(next_state,future_supply[st_offset:st_offset+self.cp_time]) # add future supply
                # self._centralized_state[query_idx] = cent_state
                next_cent_states.append(next_cent_state)
                next_durations.append(next_cs_durations)
        self._event_update[t] = []

        return len(states), (states, cent_states, joint_actions, next_states, next_cent_states, rewards, dones, durations, next_durations, etc)

