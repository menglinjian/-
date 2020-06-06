import numpy as np
import collections
import torch
import os
import pickle
import copy
from rlcard.utils.utils import *
import random
import multiprocessing
from multiprocessing import Manager, Pool, Process, Queue
#import psutil
import time
from rlcard.utils.utils import set_global_seed
import math

class DeepCFRagent():
    def __init__(self, 
                    env,
                    epochs = 1, # NN epochs
                    mini_batches = 1,
                    CFR_num = 1,
                    tra_num = 1, 
                    startPolicy = 10,
                    robustSample = False, 
                    sample_num = 5,
                    sample_rate = 0.5,
                    is_training = True, 
                    memory_size = 20000, 
                    batch_size = 2, 
                    model_path='./DeepCFR_model'):
        torch.set_num_threads(1) #设置线程为1, 不可改
        self.env = env
        self.model_path = model_path
        self.is_training = is_training
        self.CFR_num = CFR_num
        self.tra_num = tra_num
        self.sample_num = sample_num
        self.robustSample = robustSample
        self.sample_rate = sample_rate
        self.startPolicy = startPolicy
        self.action_num = self.env.action_num
        # A policy is a dict state_str -> action probabilities
        self.policy = collections.defaultdict(list)
        self.average_policy =collections.defaultdict(np.array)

        # Regret is a dict state_str -> action regrets
        self.regrets = collections.defaultdict(np.array)
        self.iteration = 0
        self.now_obs=[]

        #network
        self.regretNN = CFREstimator(self.action_num,epochs=epochs,mini_batches=mini_batches,isRegret=True,batch_size=batch_size)
        self.regretMemory = Memory(self.env.card2index, memory_size=memory_size, batch_size=batch_size)
        self.policyNN = CFREstimator(self.action_num, epochs=epochs, mini_batches=mini_batches,isRegret=False,batch_size=batch_size)
        self.policyMemory = Memory(self.env.card2index, memory_size=memory_size, batch_size=batch_size)

    def clean_Memory(self):
        self.regretMemory.clean()
        self.policyMemory.clean()

    def deepCFR(self, i, Process_num=16):
        import time 
        costtime = []
        total_start = time.perf_counter()
        for t in range(0, self.CFR_num):
            self.iteration += 1
            print('iteration is ',self.iteration)
            start = time.perf_counter()

            for player_id in range(self.env.player_num):
                print('now is player', player_id)
                torch.multiprocessing.freeze_support()
                q = torch.multiprocessing.Manager().Queue()
                p = torch.multiprocessing.Pool(Process_num)
                j = Process_num
                #orch.multiprocessing.spawn(fn=self.multi_traverse, args = (q, player_id, self.tra_num), nprocs=Process_num, join=True, daemon=False)
                for k in range(Process_num):
                    p.apply_async(self.multi_traverse, args = (q, player_id, self.tra_num))
                p.close()
                p.join()
                for k in range(Process_num):
                    u = q.get()
                    r = u[0]
                    pl = u[1]
                    for item in r:
                        self.feed(item, player_id, isRegret=True)
                    for item in pl:
                        self.feed(item, player_id, isRegret=False)
                self.train(player_id, isRegret=True)

            self.train(isRegret=False)
            end = time.perf_counter()
            print('iteration {} cost {}s'.format(self.iteration, end - start))
            costtime.append(end - start)

        total_end = time.perf_counter()
        print("epoch {} cost {}s, average is {}s".format(i+1, total_end - total_start, np.mean(costtime)))

    def train(self, player_id=0, isRegret=True):
        if isRegret==True:
            print("Now is regret training for player{}".format(player_id))
            self.regretNN.train(self.regretMemory, self.iteration)
        else:
            print("Now is policy training")
            self.policyNN.train(self.policyMemory, self.iteration)
        print('End this train')

    def multi_traverse(self, q, player_id, num):
        regretMemory = []
        policyMemory = []
        set_global_seed(os.getpid())
        for i in range(num):
            self.env.init_game()
            probs = np.ones(self.env.player_num)
            self.traverse_tree(max(self.iteration-self.startPolicy, 0), 1, player_id, regretMemory, policyMemory)
        q.put([regretMemory, policyMemory])
        return [regretMemory, policyMemory]

    def traverse_tree(self, w, prob, player_id, regretMemory, policyMemory):
        if self.env.is_over():
            return self.env.get_payoffs()

        current_player = self.env.get_player_id()

        try:
            action_utilities = {}
            state_utility = np.zeros(self.env.player_num)
            obs, legal_actions,state = self.get_state(current_player)
            regrets = self.get_regrets(state)
            action_probs = self.regret_matching(regrets)
            action_probs = remove_illegal(action_probs, legal_actions)
        except Exception as e:
            print(str(e))

        if current_player == player_id:
            try:
                for action in legal_actions:
                    action_prob = action_probs[action]
                    self.env.step(action)
                    utility = self.traverse_tree(w, prob, player_id, regretMemory, policyMemory)
                    self.env.step_back()

                    state_utility += action_prob * utility
                    action_utilities[action] = utility[player_id]

                if random.random()<self.sample_rate:
                    for action in legal_actions:
                        regrets[action] = max(0, self.regret_constrain(w-1)*regrets[action] + 
                                                w*(action_utilities[action] - state_utility[current_player]))/self.regret_constrain(w)#*math.sqrt(w))
                    
                        s = {}
                        s['hand'] = state['hand']
                        s['public_cards'] = state['public_cards']
                        s['all_chips'] = state['all_chips']
                        s['action_history'] = state['action_history']
                        #print(s)
                        regretMemory.append([s, w, regrets])

            except Exception as e:
                print(str(e))
        else:
            try:
                for action in legal_actions:
                    action_prob = action_probs[action]

                    prob_temp = prob * action_prob
                    self.env.step(action)
                    utility = self.traverse_tree(w, prob_temp, player_id, regretMemory, policyMemory)
                    self.env.step_back()
                    state_utility += utility

                if random.random()<self.sample_rate:
                    s = {}
                    s['hand'] = state['hand']
                    s['public_cards'] = state['public_cards']
                    s['all_chips'] = state['all_chips']
                    s['action_history'] = state['action_history']
                    last_action_probs = self.get_last_policy(state)
                    if w==1:
                        action_probs = action_probs*prob
                    else:
                        action_probs = (action_probs*prob*w*w + self.policy_constrain(w-1)*last_action_probs)/self.policy_constrain(w)
                    policyMemory.append([s, w, action_probs])

            except Exception as e:
                print(str(e))

        return state_utility
    
    def regret_constrain(self, t):
        return 1#t+self.startPolicy

    def policy_constrain(self, t):
        if t<=0:
            return 1
        return t*(t*t+1)/3

    def get_last_policy(self, state):
        x_test = self.state_to_array(state)
        chips = np.array(state['all_chips']).reshape(2,1)
        temp = state['action_history'].reshape(2,8)
        chips = np.concatenate((temp, chips),axis=1)
        x_test = [torch.torch.from_numpy(x_test).unsqueeze(0).float(), torch.torch.from_numpy(chips).unsqueeze(0).unsqueeze(0).float()]
        action_probs = self.policyNN.predict(x_test)[0].numpy()
        return action_probs

    def state_to_array(self, state):
        Range = np.zeros(4*13)
        Range_self = np.zeros(4*13)
        Range_public = np.zeros(4*13)
        idx_self = [self.env.card2index[card] for card in state['hand']]
        idx_public = [self.env.card2index[card] for card in state['public_cards']]

        Range[idx_self] = 1
        Range[idx_public] = 1
        Range_self[idx_self] = 1
        Range_public[idx_public] = 1
        Range = Range.reshape((1,4,13))
        Range_self = Range_self.reshape((1,4,13))
        Range_public = Range_public.reshape((1,4,13))

        x_test = np.concatenate([Range,Range_self,Range_public])
        return x_test
    
    def get_regrets(self, state):
        try:
            x_test = self.state_to_array(state)
            chips = np.array(state['all_chips']).reshape(2,1)
            temp = state['action_history'].reshape(2,8)
            chips = np.concatenate((temp, chips),axis=1)
            x_test = [torch.torch.from_numpy(x_test).unsqueeze(0).float(), torch.torch.from_numpy(chips).unsqueeze(0).unsqueeze(0).float()]
            regrets = self.regretNN.predict(x_test)[0].numpy()
        except Exception as e:
            print(str(e))
        return regrets

    def get_policy(self, state, legal_actions, isEval=False):
        x_test = self.state_to_array(state)
        chips = np.array(state['all_chips']).reshape(2,1)
        temp = np.array(state['action_history']).reshape(2,8)
        chips = np.concatenate((temp, chips),axis=1)
        x_test = [torch.torch.from_numpy(x_test).unsqueeze(0).float(), torch.torch.from_numpy(chips).unsqueeze(0).unsqueeze(0).float()]
        if not isEval:
            #regret matching to get policy
            regret = self.regretNN.predict(x_test)[0].numpy()
            action_probs = self.regret_matching(regret)
            #action_probs = self.policyNN.predict(x_test)[0].numpy()
        else:
            action_probs = self.policyNN.predict(x_test)[0].numpy()

        if legal_actions==None:
            if np.sum(action_probs) == 0:
                action_probs[legal_actions] = 1 / len(legal_actions)
            else:
                action_probs /= sum(action_probs)
        else:
            action_probs = remove_illegal(action_probs, legal_actions)

        action_probs = remove_illegal(action_probs, legal_actions)
        return action_probs

    def update_policy(self):
        ''' Update policy based on the current regrets
        '''
        for obs in self.now_obs:
            self.policy[obs] = self.regret_matching(obs)
        self.now_obs = []

    def regret_matching(self, regret):#need to change
        ''' Apply regret matching

        Args:
            regret (tensor): The regrets
        '''
        #regret = self.regrets[obs]
        positive_regret_sum = sum([r for r in regret if r > 0])

        action_probs = np.zeros(self.env.action_num)
        if positive_regret_sum > 0:
            for action in range(self.env.action_num):
                action_probs[action] = max(0.0, regret[action] / positive_regret_sum)
        else:
            for action in range(self.env.action_num):
                action_probs[action] = 1.0 / self.env.action_num

        return action_probs

    def eval_step(self, state):
        legal_actions = state['legal_actions']
        action_probs = self.get_policy(state, legal_actions, isEval=True)
        action = np.random.choice(len(action_probs), p=action_probs)
        return action

    def feed(self, state, player_id, isRegret=True):
        #state [hand, public_cards, chips(action probs)]
        #[['H3', 'H6'], ['DK', 'D9'], [1,2]]
        if isRegret==True:
            self.regretMemory.save(state)
        else:
            self.policyMemory.save(state)

    def get_state(self, player_id):
        ''' Get state_str of the player

        Args:
            player_id (int): The player id

        Returns:
            (tuple) that contains:
                state (str): The state str
                legal_actions (list): Indices of legal actions
        '''
        state = self.env.get_state(player_id)
        obs = self.state_to_obs(state)
        #print(state['round'], state['current_player'], '\n',state['action_history'])
        return obs.tostring(), state['legal_actions'], state

    def state_to_obs(self, state):
        public_cards = state['public_cards']
        hand = state['hand']
        idx = [self.env.card2index[card] for card in public_cards]
        obs = np.zeros(8)
        obs[idx] = 1
        idx = [self.env.card2index[card] for card in hand]
        obs[idx] = 2
        obs[6] = state['all_chips'][0]
        obs[7] = state['all_chips'][1]
        return obs
    
    def compute_exploitability(self, eval_num):
        self.cards = [Card('S', 'J'), Card('H', 'J'), Card('S', 'Q'), Card('H', 'Q'), Card('S', 'K'), Card('H', 'K')]
        self.env.init_game()
        palyers_range=[]
        Range = []
        
        for i in range(0,len(self.cards)):
            Range.append([self.cards[i], 1/6])

        for player_id in range(self.env.player_num):
            palyers_range.append(Range)
        
        utility = []
        for i in range(0, eval_num):
            for player_id in range(self.env.player_num):
                self.t = 0
                self.env.init_game()
                #print(palyers_range[0][0:3])
                palyers_range_ = copy.deepcopy(palyers_range)
                self.oppo_last_action = -1
                payoffs = self.traverse_exp(player_id, palyers_range_)
                #print("final",palyers_range[0][0:3])
                #print('player:{},payoff:{}'.format(player_id,payoffs[player_id]))
                #exit()
                utility.append(payoffs[player_id])
        
        return np.mean(utility)

    def traverse_exp(self, player_id, palyers_range):
        self.t=self.t+1
        if self.env.is_over():
            #print(self.env.get_payoffs())
            return self.env.get_payoffs()
        temp = 0
        current_player = self.env.get_player_id()
        _, legal_actions, state = self.get_state(current_player)
        #print('current_player:{}, legal_actions:{}'.format(current_player, legal_actions))
        #print("in",palyers_range[current_player][0:3])
        if current_player != player_id:
            action_probs = self.get_policy(state, legal_actions, isEval=True)
            action = np.random.choice(len(action_probs), p=action_probs)
            self.oppo_last_action = action
            self.oppo_last_legal_actions = legal_actions
            self.oppo_last_state = state
            self.env.step(action)

        else:
            #更新对手range
            Sum = 0
            handcards = state['hand']
            publiccards = state['public_cards']
            for i,obj in enumerate(palyers_range[1-current_player]):
                oppocards = [obj[0].get_index()]
                if len(handcards+oppocards)!=len(set(handcards+oppocards)) or len(publiccards+oppocards)!=len(set(publiccards+oppocards)):
                    palyers_range[1-current_player][i][-1] = 0
                    temp = i
                else:
                    if self.oppo_last_action == -1:
                        action_prob = 1
                    else:
                        oppo_state = self.oppo_last_state
                        oppo_state['hand'] = oppocards
                        action_prob = self.get_policy(oppo_state, self.oppo_last_legal_actions, isEval=True)[self.oppo_last_action]
                    palyers_range[1-current_player][i][-1] = palyers_range[1-current_player][i][-1]*action_prob
                    Sum = Sum + palyers_range[1-current_player][i][-1]

            for i,obj in enumerate(palyers_range[1-current_player]):
                palyers_range[1-current_player][i][-1] = palyers_range[1-current_player][i][-1]/Sum

            action = self.LocalBR(player_id, state, legal_actions, palyers_range[1-current_player])#对两人情况
            self.env.step(action)
        
        #print("out",palyers_range[current_player][0:3],"---",current_player, palyers_range[current_player][temp])
        #print("times:{},palyer:{},action:{}".format(self.t,current_player,action))
        return self.traverse_exp(player_id, palyers_range)

    def LocalBR(self, player_id, state, legal_actions, oppo_range):
        values = np.zeros(self.env.action_num)
        handcards = state['hand']
        publiccards = state['public_cards']
        pot_myself = state['all_chips'][0]
        pot_oppo = state['all_chips'][1]

        wp = self.WpRollout(player_id, handcards, publiccards, oppo_range)
        asked = pot_oppo - pot_myself 
        #print('asked',asked)
        values[0] = wp*pot_myself-(1-wp)*asked
        for action in legal_actions:
            if action >=2 :
                fp = 0
                oppo_range_temp = copy.deepcopy(oppo_range)
                Sum = 0
                self.env.step(action)
                _, oppo_legal_actions,state_ = self.get_state(1-player_id)
                #print(action, oppo_legal_actions)
                diff = state_['all_chips'][1] - pot_myself
                oppo_state = state_
                #print(state_['all_chips'],pot_myself+action-2)
                for i,obj in enumerate(oppo_range_temp):
                    oppocards = [obj[0].get_index()]
                    prob = obj[-1]
                    if len(handcards+oppocards)!=len(set(handcards+oppocards)) or len(publiccards+oppocards)!=len(set(publiccards+oppocards)):
                        oppo_range_temp[i][-1] = 0
                    else:
                        oppo_state['hand'] = oppocards
                        foldprob = self.get_policy(oppo_state, oppo_legal_actions, isEval=True)[1]
                        fp = fp + prob*foldprob
                        oppo_range_temp[i][-1] = oppo_range_temp[i][-1]*(1-foldprob)
                        Sum = Sum + oppo_range_temp[i][-1]
                for i,obj in enumerate(oppo_range_temp):
                    oppo_range_temp[i][-1] = oppo_range_temp[i][-1]/Sum
                
                self.env.step_back()

                wp = self.WpRollout(player_id, handcards, publiccards, oppo_range_temp)
                values[action] = fp*pot_myself + (1-fp)*(wp*(pot_myself+diff)-(1-wp)*(asked+diff))
                #print("fp",fp," Sum",Sum,'value',values[action])
                #print("action:",action,'--',values[action])
        result = np.argmax(values)
        if values[result]>0:
            return result
        else:
            return 1#flod

    def WpRollout(self, player_id, handcards, publiccards, oppo_range):
        from rlcard.games.limitholdem.utils import compare_hands
        wp = 0 
        handcards = handcards
        publiccards = publiccards
        for i,obj in enumerate(oppo_range):
            oppocards = [obj[0].get_index()]
            prob = obj[-1]
            if prob!=0:
                if len(handcards+oppocards)!=len(set(handcards+oppocards)) or len(publiccards+oppocards)!=len(set(publiccards+oppocards)):
                    pass
                else:
                    cards = self.cards
                    #print('enter')
                    if len(publiccards)<1:
                        cards = [i.get_index() for i in cards if (i.get_index() not in handcards and i.get_index() not in oppocards)]
                        for c in cards:
                            #print(c)
                            publiccards_temp = publiccards + [c]
                            selfcards_temp = handcards 
                            oppocards_temp = oppocards
                            wp = wp + 0.25*prob*self.compare_hands(selfcards_temp, oppocards_temp, publiccards_temp)[0]
                    else:
                        publiccards_temp = publiccards
                        selfcards_temp = handcards 
                        oppocards_temp = oppocards 
                        wp = wp + prob*self.compare_hands(selfcards_temp, oppocards_temp, publiccards_temp)[0]
        return wp

    def compare_hands(self, selfcards, oppocards, publiccards):
        from rlcard.utils.utils import rank2int
        # Judge who are the winners
        winners = [0, 0]
        if sum(winners) < 1:
            if selfcards[0][1] == oppocards[0][1]:
                winners = [1, 1]
        if sum(winners) < 1:
            if selfcards[0][1] == publiccards[0][1]:
                winners[0] = 1
            elif oppocards[0][1] == publiccards[0][1]:
                winners[1] = 1
        if sum(winners) < 1:
            winners = [1, 0] if rank2int(selfcards[0][1]) > rank2int(oppocards[0][1]) else [0, 1]
        return winners

    def save(self):
        self.regretNN.save(self.model_path+'regretmodel'+'.pth')
        self.policyNN.save(self.model_path+'policyodel.pth')

    def load(self):
        self.regretNN.load(self.model_path+'regretmodel'+'.pth')
        self.policyNN.load(self.model_path+'policyodel.pth')



import torch
import torch.nn as nn
import torch.nn.functional as F
class CFREstimator():
    def __init__(self,
                outsize, 
                epochs = 100, 
                mini_batches = 1000,
                batch_size = 8,
                isRegret=False,
                device = None):
        self.outsize = outsize
        self.device = device
        self.epochs = epochs
        self.mini_batches = mini_batches
        self.isRegret = isRegret
        self.batch_size = batch_size
        self.Net = CFRNetwork(outsize, isRegret=isRegret)
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        #self.Net.to(self.device)
        #self.Net.share_memory()
        #print(self.Net)

    def train(self, Memory, iteration):
        import torch.optim as optim
        self.Net.to(self.device)
        torch.set_num_threads(8)
        criterion = tloss(self.device).to(self.device)
        optimizer = optim.Adam(self.Net.parameters(), lr=0.0001)
        for epoch in range(0,self.epochs):
            running_loss = 0.0
            for i in range(0,self.mini_batches):
                t, x_train, chips, y_train = Memory.sample()
                x_train = [torch.torch.from_numpy(x_train).float().to(self.device), 
                            torch.torch.from_numpy(chips).float().to(self.device)]
                y_train = torch.torch.from_numpy(y_train).float().to(self.device)
                optimizer.zero_grad()
                outputs = self.Net(x_train).to(self.device)
                
                loss = criterion(outputs, y_train, t, iteration, self.isRegret)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            if (epoch+1)%50==0 or epoch == 0:
                if self.isRegret:
                    print('%dth epoch loss: %.3f' %(epoch + 1,running_loss / self.mini_batches))
                else:
                    print('%dth epoch loss: %.3f' %(epoch + 1,10000*running_loss / self.mini_batches))
        torch.set_num_threads(1)
        self.Net.to(torch.device('cpu'))

    def predict(self,input):
        with torch.no_grad():
            try:
                output = self.Net(input) #x:batch_size*3*4*13, chips:batch_size*1*2*9
            except Exception as e:
                print(str(e))
        return output

    def save(self, path):
        self.path = path
        torch.save(self.Net.state_dict(), path)
    
    def load(self, path = None):
        if path==None:
            path = self.path
        self.Net.load_state_dict(torch.load(path))
        

class tloss(nn.Module):
    def __init__(self, device):
        super(tloss, self).__init__()
        self.device = device

    def forward(self, pred, target, t, iteration, isRegret = True):
        size = pred.size()[1]
        pred = pred.view(-1)
        target = target.view(-1)
        t= t.reshape(t.shape[0],1)
        t = torch.from_numpy(t.repeat(size, axis = 1)).view(-1).to(self.device)
        if isRegret:
            loss = torch.mean(torch.pow(pred-target,2))
        else:
            loss = torch.mean(torch.pow(t,4)*torch.pow(pred-target,2))
            #print(torch.pow(pred-target,2))
        return loss

class SpatialAttention(nn.Module):#CBAM
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, 3, padding=1)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, input):
        try:
            x = torch.cat((torch.max(input, 1)[0].unsqueeze(1), torch.mean(input, 1).unsqueeze(1)), dim=1)
            x = F.sigmoid(self.bn(self.conv(x)))
            x = input*x
        except Exception as e:
            print(str(e))
        return x

class ChannelAttention(nn.Module):
    def __init__(self):
        super(ChannelAttention,self).__init__()
        self.fc1 = nn.Linear(32, 4)
        self.fc2 = nn.Linear(4, 32)

    def forward(self, input):
        try:
            x = F.max_pool2d(input, input.shape[2])
            x = x.view(-1, self.num_flat_features(x))
            x = F.relu(self.fc1(x))
            x = F.sigmoid(self.fc2(x)).unsqueeze(2).unsqueeze(3).expand_as(input)
            x = x*input
        except Exception as e:
            print(str(e))
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class AttentionResNet(nn.Module):
    def __init__(self):
        super(AttentionResNet,self).__init__()
        self.Sattention = SpatialAttention()
        self.Cattention = ChannelAttention()

    def forward(self, input):
        x = self.Sattention(input)
        x = self.Cattention(x)
        x = F.relu(input+x)
        return x

class ConvbatchNet(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=2, stride=1, padding=0):
        super(ConvbatchNet,self).__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(output_channel)

    def forward(self, input):
        x = self.conv(input)
        x = self.bn(x)
        x = F.relu(x)
        return x

class CFRNetwork(nn.Module):
    def __init__(self,
                outsize, 
                epochs=100,
                isRegret=False):
        super(CFRNetwork,self).__init__()
        self.isRegret = isRegret
        self.padding1 = nn.ReplicationPad2d((2,2,6,7))
        self.conv1 = nn.Conv2d(3, 6, 1)
        self.conv2 = ConvbatchNet(9, 16, 3)
        self.conv3 = ConvbatchNet(16, 32, 3)
        self.conv4 = ConvbatchNet(32, 32, 2, padding=1)
        self.fc1 = nn.Linear(146, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, outsize)
        self.at3 = AttentionResNet()
        self.at4 = AttentionResNet()

    def forward(self, input):
        import threading
        #print(print(threading.currentThread().ident),'forward')
        x, chips = input[0], input[1]# x:batch_size*3*4*13, chips:batch_size*1*2*9
        try:
            #print(chips.view(-1, self.num_flat_features(chips)))
            #print(F.relu(x))
            x = self.padding1(x)
            x_ = F.relu(self.conv1(x))
            x = torch.cat((x, x_), 1)
            x = self.conv2(x)
            x = F.max_pool2d(x, 2)
            x = self.conv3(x)
            x = self.at3(x)
            x = self.conv4(x)
            x = self.at4(x)
            x = F.max_pool2d(x, 3)
            x = x.view(-1, self.num_flat_features(x))#torch.Size([5, 128])
            x = torch.cat((x, chips.view(-1, self.num_flat_features(chips))), 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            if self.isRegret == False:
                x = F.relu(self.fc4(x))
            else:
                x = F.relu(self.fc4(x))
        except Exception as e:
            print(str(e))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Memory(object):
    ''' Memory for saving transitions
    '''

    def __init__(self, card2index, memory_size, batch_size):
        ''' Initialize
        Args:
            memory_size (int): the size of the memroy buffer
        '''
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []
        self.card2index = card2index

    def state_to_array(self, state):
        Range = np.zeros(4*13)
        Range_self = np.zeros(4*13)
        Range_public = np.zeros(4*13)
        idx_self = [self.card2index[card] for card in state['hand']]
        idx_public = [self.card2index[card] for card in state['public_cards']]

        Range[idx_self] = 1
        Range[idx_public] = 1
        Range_self[idx_self] = 1
        Range_public[idx_public] = 1
        Range = Range.reshape((1,4,13))
        Range_self = Range_self.reshape((1,4,13))
        Range_public = Range_public.reshape((1,4,13))

        x_test = np.concatenate([Range,Range_self,Range_public])
        return x_test

    def save(self, state):
        ''' Save transition into memory

        Args:
            # [{'hand': ['HJ'], 'public_cards': ['SJ'], 'all_chips': [10, 10]},t,array()]
            agent.feed([{hand, public_cards, chips}, self.iteration, policy(regert)])
        '''
        state ,t, y_train = state
        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        chips = np.array(state['all_chips']).reshape(2,1)
        temp = state['action_history'].reshape(2,8)
        chips = np.concatenate((temp, chips),axis=1)
        x_train = self.state_to_array(state)
        self.memory.append([t,x_train,chips,y_train])

    def sample(self):
        ''' 
        Returns:
            t (int): iteration
            x_train (np.array): a batch of states
            chips(np.array):
            y_train (list): reward
        '''
        #print(len(self.memory))
        samples = random.sample(self.memory, self.batch_size)
        return map(np.array, zip(*samples))
    
    def clean(self):
        self.memory = []
