import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
from torch.distributions import Categorical, Normal
from collections import deque
from net import ReplayBuffer, Actor, Critic, OUNoise

TorchFloat = None
TorchLong = None

class Agent_MASTER(nn.Module):
    def __init__(self, env, args, LOAD_PATH=None):
        """Initialization.
        Args:
            env (gym.Env): Charging environment
            args
        """
        super(Agent_MASTER, self).__init__()
        global TorchFloat,TorchLong
        TorchFloat = torch.cuda.FloatTensor if args.device == torch.device('cuda') else torch.FloatTensor
        TorchLong = torch.cuda.LongTensor if args.device == torch.device('cuda') else torch.LongTensor

        self.env = env
        self.device = args.device
        self.gamma, self.lr_a, self.lr_c = args.gamma, args.lr_a, args.lr_c
        self.soft_tau_a, self.soft_tau_c = args.soft_tau_a, args.soft_tau_c
        self.T_LEN, self.N, self.action_dim = args.T_LEN, args.N, args.action_dim
        self.n_pred, self.miss_time, self.interval = args.n_pred, args.miss_time, args.interval
        self.T = self.T_LEN*self.interval
        self.limit_waiting_time = args.n_pred*args.interval
        self.clip_norm = args.clip_norm
        self.test = args.test
        self.batch_size = args.batch_size
        self.noise = args.noise
        self.replay_buffer = ReplayBuffer(args.replay_buffer_size)
        self.ouNoise = OUNoise(args)
        # self.dropout = nn.Dropout(0.9)
        self.acr = args.ac_ratio
        self.n_spa = args.n_spa
        self.temp = args.temp
        
        """ optimal critic """
        ## CWT optimal
        self.Critic_cwt_op = Critic(args).to(self.device).eval()
        self.Actor_cwt_op = Actor(args).to(self.device).eval()
        if(not args.simulate):
            CWT_OP_PATH = "../master_oneob/params/oneob_CWT_optimal.pkl"
            print("Loading parameters: {}".format(CWT_OP_PATH))
            cwt_state_dict = torch.load(CWT_OP_PATH) ### should be pre-trained for pratical use 
            self.Critic_cwt_op.load_state_dict(cwt_state_dict['critic'])
            self.Actor_cwt_op.load_state_dict(cwt_state_dict['actor'])
        
        ### CP optimal
        self.Critic_cp_op = Critic(args).to(self.device).eval()
        self.Actor_cp_op = Actor(args).to(self.device).eval()
        if(not args.simulate):
            CP_OP_PATH = "../stddpg_oneob/params/oneob_CP_optimal.pkl"
            print("Loading parameters: {}".format(CP_OP_PATH))
            cp_state_dict = torch.load(CP_OP_PATH) ### should be pre-trained for pratical use 
            self.Critic_cp_op.load_state_dict(cp_state_dict['critic'])
            self.Actor_cp_op.load_state_dict(cp_state_dict['actor'])
        
        ### networks: Critic (Q_funciton), Actor
        # cwt critic
        self.Critic_cwt = Critic(args).to(self.device)
        if(args.load == True):
            state_dict = torch.load(LOAD_PATH)
            self.Critic_cwt.load_state_dict(state_dict['critic_cwt'])
        self.Critic_target_cwt = Critic(args).to(self.device)
        self.Critic_target_cwt.load_state_dict(self.Critic_cwt.state_dict())
        self.Critic_target_cwt.eval()
        
        # fee critic
        self.Critic_fee = Critic(args).to(self.device)
        if(args.load == True):
            self.Critic_fee.load_state_dict(state_dict['critic_fee'])
        self.Critic_target_fee = Critic(args).to(self.device)
        self.Critic_target_fee.load_state_dict(self.Critic_fee.state_dict())
        self.Critic_target_fee.eval()

        ### actor
        self.Actor = Actor(args).to(self.device)
        if(args.load == True):
            self.Actor.load_state_dict(state_dict['actor'])
            print("Loading parameters: {}".format(LOAD_PATH))
        self.Actor_target = Actor(args).to(self.device)
        self.Actor_target.load_state_dict(self.Actor.state_dict())
        self.Actor_target.eval()
        
        # loss and optimizer
        self.mseloss = torch.nn.MSELoss()
        self.optimizer_critic_cwt = torch.optim.Adam(self.Critic_cwt.parameters(),\
                     lr=args.lr_c, betas=(0.9, 0.99), eps=1e-5)
        self.optimizer_critic_fee = torch.optim.Adam(self.Critic_fee.parameters(),\
                     lr=args.lr_c, betas=(0.9, 0.99), eps=1e-5)
        self.optimizer_actor = torch.optim.Adam(self.Actor.parameters(),\
                     lr=args.lr_a, betas=(0.9, 0.99), eps=1e-5)
    
    def reset_agent(self):
        self.LASTUPDATE_T = -1

    def adjust_learning_rate(self, optimizer, lr):
        optimizer.param_groups[0]["lr"] = lr
        return lr

    def stack_transition(self, transition, is_tensor=True):
        np_trans = []
        for ele in transition:
            if(is_tensor):
                np_trans.append(torch.cat(ele,dim=0))
            else:
                np_trans.append(np.stack(ele))
        return np_trans

    def state_normalization(self, state):
        power = state[...,:1].div(150)
        supply = state[...,1:2]
        supply_clip = 10
        supply = torch.where(supply<=supply_clip, supply, supply_clip*torch.ones(1,1).to(self.device)).div(supply_clip)
        demand = state[...,2:3].div(20)
        t_step = state[...,3:4]
        chargefee = state[...,4:5].div(2.3)
        cs_idxs = state[...,5:6]
        duration = state[...,6:7].div(self.miss_time)

        # (t_step,supply,demand,duration)
        norm_state = torch.cat([t_step,cs_idxs,supply,demand,power,chargefee,duration],dim=-1) # (K, F)
        return norm_state
    
    def Centralized_state_action(self, cent_states, joint_actions, durations, inds=None):
        B,N,_ = cent_states.shape
        state_actions = torch.cat([cent_states, joint_actions],dim=-1)
        state_actions, inds = self.env.get_observe_torch(state_actions, durations, inds, self.n_spa)
        return state_actions, inds
    
    def action_estimation(self, t_querys, t, query_idxs, norm_state, duration, n_iter, is_test=False):
        """ policy-based estimation
        """
        n_q = len(t_querys)
        nq_rec = 0
        global_actions = self.Actor(norm_state).detach()
        if((not self.test) and (not is_test)):
            global_actions = self.ouNoise.action_noise(global_actions, n_iter)
        actions, inds = self.env.get_observe_torch(global_actions, duration) # (n_q, k, 1)
        assert not torch.isnan(actions).any()
        action_cs_idx = []
        original_cs_idx = []
        is_rec = np.full(n_q, False)
        for i, query_tp in enumerate(t_querys):
            # print(query_tp)
            o_cs_idx = query_tp[-1]
            original_cs_idx.append(o_cs_idx)
            if(np.random.rand()<=self.acr):
                is_rec[i] = True
                nq_rec += 1
                action_k_idx = torch.argmax(actions[i], dim=-2) # (n_q,1,1)
                action_cs_idx.append(inds[i,action_k_idx[0]].item()) # (n_q,)
            else:
                action_cs_idx.append(o_cs_idx)
        action_cs_idx = torch.from_numpy(np.asarray(action_cs_idx)).type(TorchLong)
        for i, q_idx in enumerate(query_idxs):
            self.env._integrated_state[q_idx] = norm_state[i].unsqueeze(dim=0)
        return action_cs_idx, nq_rec, is_rec, original_cs_idx, actions, global_actions
    
    def step(self, cur_t, n_iter, is_test=False): # one time_step
        if(cur_t==0): self.ouNoise.reset()
        losses_critic, losses_actor,rec_rewards = [],[],[]
        fee_costs,save_costs,time_costs= [],[],[]
        n_query, n_rec, n_success_charge, n_success_charge_rec = 0,0,0,0
        st_minute = cur_t*self.interval
        ed_minute = st_minute+self.interval
        if(cur_t == self.T_LEN-1): ed_minute += self.limit_waiting_time
        for t in range(st_minute, ed_minute):
            """ event1: dispose vehicle arrival at t，
            """
            t_feecost, t_save_cost, t_time_cost, success_cnt, success_cnt_rec, t_reward = self.env.arrival_step(t) # (n_q,)
            fee_costs.extend(t_feecost)
            save_costs.extend(t_save_cost)
            time_costs.extend(t_time_cost)
            rec_rewards.extend(t_reward)
            n_success_charge += success_cnt
            n_success_charge_rec += success_cnt_rec
            if(t>=self.T and t!=self.T+self.limit_waiting_time-1): continue
            """ event2: dispose charging query at t
            """
            t_querys = self.env.get_query(t)
            n_q = len(t_querys)
            n_query += n_q
            if(n_q > 0 and t<self.T):
                t_state = torch.from_numpy(self.env.get_state(t)).type(TorchFloat).repeat(n_q,1,1) # (N,F)
                duration = np.asarray([self.env.grid2allcs_durations(query_tp[0]) for query_tp in t_querys]) # (n_q,N,1) 
                nq_duration = torch.from_numpy(duration).type(TorchFloat)
                primal_state = torch.cat([t_state,nq_duration],dim=-1) #(n_q,N,F)
                norm_state = self.state_normalization(primal_state) #(n_q,F) 
                query_idxs = [t_querys[i][2] for i in range(n_q)]
                ### take actions
                action_cs_idx, nq_rec, is_rec, original_cs_idx, actions, global_actions = self.action_estimation(t_querys, t, query_idxs, norm_state, duration, n_iter, is_test) # (n_q,)
                n_rec += nq_rec
                self.env.query_step(n_q, t_querys, action_cs_idx, actions, global_actions, is_rec, original_cs_idx)

            """ *** query indexing *** 
            """
            if(n_q > 0 and not is_test):
                self.env.index_step(t, n_q, t_querys, self.LASTUPDATE_T)
                self.LASTUPDATE_T = t

            """ *** derive transition *** 
            """
            if(not is_test):
                n_trans, transitions = self.env.transition_step(t)
                if(n_trans>0):
                    ### add to replay buffer
                    self.replay_buffer.push(transitions)
                    
            """ *** model update *** 
            """
            if(len(self.replay_buffer) >= self.batch_size and not is_test):
                states, cent_states, joint_actions, next_states, next_cent_states, rewards, dones, durations, next_durations, etc \
                    = self.replay_buffer.sample(self.batch_size) # (batch_sizz,)
                ###########  =======================================  ##############   
                states, cent_states, joint_actions, next_states, next_cent_states = self.stack_transition\
                            ((states, cent_states, joint_actions, next_states, next_cent_states),is_tensor=True)
                rewards, dones, durations, next_durations, etc = self.stack_transition((rewards, dones, durations, next_durations, etc),is_tensor=False)

                rewards = torch.from_numpy(rewards).type(TorchFloat).view(-1,2) # (B,2)
                rewards = torch.clamp(rewards,0,10)
                dones = torch.from_numpy(dones).type(TorchFloat).view(-1,1)
                etc = torch.from_numpy(etc).type(TorchFloat)
                time_exponent = etc[:,0].view(-1,1) # (B,)

                ### Critic
                state_action, inds = self.Centralized_state_action(cent_states, joint_actions, durations) # (B,F)
                # print(state_action.shape)
                q_values_cwt = self.Critic_cwt(state_action) # (B,1)
                q_values_fee = self.Critic_fee(state_action) # (B,1)
                next_joint_actions = self.Actor_target(next_states).detach()
                next_state_action, _ = self.Centralized_state_action(next_cent_states, next_joint_actions, next_durations) # (B,K,F)
                next_q_values_cwt = self.Critic_target_cwt(next_state_action).detach()
                next_q_values_cwt = torch.clamp(next_q_values_cwt,0,100)
                next_q_values_fee = self.Critic_target_fee(next_state_action).detach()
                next_q_values_fee = torch.clamp(next_q_values_fee,0,100)
                # if(np.random.random() > 0.999):
                #     print(rewards.shape, next_q_values.shape, time_exponent.shape,dones.shape)
                discount = torch.pow(self.gamma, time_exponent)
                expected_returns_cwt = rewards[:,0].unsqueeze(dim=-1) + discount*next_q_values_cwt*(1-dones)
                expected_returns_fee = rewards[:,1].unsqueeze(dim=-1) + discount*next_q_values_fee*(1-dones)
                # Critic parameters update
                critic_loss_item_cwt = self.update_critic_cwt(q_values_cwt, expected_returns_cwt)
                critic_loss_item_fee = self.update_critic_fee(q_values_fee, expected_returns_fee)
                losses_critic.append((critic_loss_item_cwt+critic_loss_item_fee)/2)

                ### Actor
                new_joint_actions = self.Actor(states) # (B,N,1)
                new_joint_actions_cwt = self.Actor_cwt_op(states) 
                new_joint_actions_cp = self.Actor_cp_op(states)
                new_state_actions, _ = self.Centralized_state_action(cent_states, new_joint_actions, durations, inds) # (B,K,F)
                new_state_actions_cwt, _ = self.Centralized_state_action(cent_states, new_joint_actions_cwt, durations, inds) # (B,K,F)
                new_state_actions_cp, _ = self.Centralized_state_action(cent_states, new_joint_actions_cp, durations, inds) # (B,K,F)
                topk_actions, _ = self.env.get_observe_torch(new_joint_actions, durations, inds, self.n_spa) # (B,k,1)
                critic_value_cwt = self.Critic_cwt(new_state_actions) # (B,1)
                critic_value_cwt_op = self.Critic_cwt_op(new_state_actions_cwt)*1.2 # (B,1)
                critic_value_fee = self.Critic_fee(new_state_actions) # (B,1)
                critic_value_cp_op = self.Critic_cp_op(new_state_actions_cp)*1.2 # (B,1)
                
                weight_cwt = (critic_value_cwt_op-critic_value_cwt)/critic_value_cwt_op
                weight_cp = (critic_value_cp_op-critic_value_fee)/critic_value_cp_op

                score = torch.softmax(torch.cat([weight_cwt.mul(self.temp),weight_cp.mul(self.temp)],dim=-1),dim=-1)
                avg_critic_value = 2*critic_value_cwt*score[:,0] + 2*critic_value_fee*score[:,1] # (B,)
                actor_loss = - avg_critic_value.mean() + 0.1*torch.mean(torch.pow(topk_actions,2))
                    
                ### Actor parameters update
                actor_loss_item = self.update_actor(actor_loss) 
                losses_actor.append(actor_loss_item)
                self.soft_update(self.Critic_target_cwt, self.Critic_cwt, self.soft_tau_c)
                self.soft_update(self.Critic_target_fee, self.Critic_fee, self.soft_tau_c)
                self.soft_update(self.Actor_target, self.Actor, self.soft_tau_a)

        return fee_costs, save_costs, time_costs, losses_critic, losses_actor, n_query, n_rec, n_success_charge, n_success_charge_rec, rec_rewards
    
    def update_critic_cwt(self, q_values, target_q_values):
        """Update critic params by gradient descent."""
        loss = self.mseloss(q_values, target_q_values) 
        self.optimizer_critic_cwt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.Critic_cwt.parameters(), 0.5)
        self.optimizer_critic_cwt.step()
        return loss.item()

    def update_critic_fee(self, q_values, target_q_values):
        """Update critic params by gradient descent."""
        loss = self.mseloss(q_values, target_q_values) 
        self.optimizer_critic_fee.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.Critic_fee.parameters(), 0.5)
        self.optimizer_critic_fee.step()
        return loss.item()

    def update_actor(self, actor_loss):
        """Update actor params by gradient descent."""
        loss = actor_loss
        self.optimizer_actor.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.Actor.parameters(), 0.5)
        self.optimizer_actor.step()
        return loss.item()

    def soft_update(self, target, src, soft_tau):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.data.copy_(target_param.data*(1.0-soft_tau) + param.data*soft_tau)

    # def hard_update(self):
    #     """Hard update: target <- local."""
    #     self.Qtarget.load_state_dict(self.Qnet.state_dict())

