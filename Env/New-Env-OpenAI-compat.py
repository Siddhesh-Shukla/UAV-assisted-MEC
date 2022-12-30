import math
import gym
from gym import spaces
import numpy as np

class UAVEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    MAX_BATTERY_UAV = e_battery_uav = 50000000
    height = ground_length = ground_width = 100
    SUM_TASK_SIZE = curr_sum_task_size = 300 * 1048576
    loc_uav = [50, 50]
    bandwidth_nums = 1
    B = bandwidth_nums * 10 ** 6
    p_noisy_los = 10 ** (-13)
    p_noisy_nlos = 10 ** (-11)
    flight_speed = 50.
    f_ue = 6e8
    f_uav = 6e9
    r = 10 ** (-27)        
    s = 1000
    p_uplink = 0.1
    alpha0 = 1e-5
    v_ue = 1
    t_fly = 1
    t_com = 5
    m_uav = 9.65

    M = 4
    action_dim = 4
    state_dim = 3 * M + 4
    block_flag_list = np.random.randint(0, 2, M)  
    loc_ue_list = np.random.randint(0, 101, size=[M, 2])
    task_list = np.random.dirichlet(np.ones(M))*SUM_TASK_SIZE
    action_bound = [-1, 1]
    
    end_time = np.zeros(M)
    step_ = 0
    MAX_STEPS = 100

    def __init__(self):
        super(UAVEnv, self).__init__()
        self.action_space = spaces.Box( np.array([-1]*self.action_dim), np.array([1]*self.action_dim))
        self.observation_space = spaces.Box( np.array([0]*self.state_dim), 
                                            np.array([self.MAX_BATTERY_UAV, self.SUM_TASK_SIZE, 
                                                      self.SUM_TASK_SIZE, self.SUM_TASK_SIZE, 
                                                      self.SUM_TASK_SIZE, self.SUM_TASK_SIZE, 
                                                      self.ground_length, self.ground_width, 
                                                      self.ground_length, self.ground_width, 
                                                      self.ground_length, self.ground_width, 
                                                      self.ground_length, self.ground_width,
                                                      self.ground_length, self.ground_width,
                                                      ]))
        self.reset()

    def reset(self):
        self.reset_uav()
        self.reset_ue()
        return self.state

    def reset_uav(self):
        self.e_battery_uav = self.MAX_BATTERY_UAV
        self.loc_uav = [50, 50]
        self.update_state()

    def reset_ue(self):
        self.end_time = np.zeros(self.M)
        self.task_list = np.random.dirichlet(np.ones(self.M))*self.SUM_TASK_SIZE
        self.curr_sum_task_size = self.SUM_TASK_SIZE
        self.loc_ue_list = np.random.randint(0, 101, size=[self.M, 2])
        self.update_state()

    def update_state(self):
        self.state = np.append(self.e_battery_uav, self.curr_sum_task_size)
        self.state = np.append(self.state, self.task_list)
        self.state = np.append(self.state, self.loc_uav)
        self.state = np.append(self.state, self.loc_ue_list)

    def render(self, mode='human', close=False):
        print('Battery :', self.e_battery_uav, 
              '\nUAV loc :', self.loc_uav, 
              '\nTask :', self.curr_sum_task_size, self.task_list, 
              '\nUE location :', np.ravel(self.loc_ue_list))
    
    def getUENewLoc(self, x, y, dx, dy):

        new_x = (x+dx+2*self.ground_length)%2*self.ground_length
        if new_x > self.ground_length:
            new_x = 2*self.ground_length-new_x
        
        new_y = (y+dy+2*self.ground_width)%2*self.ground_width
        if new_y > self.ground_width:
            new_y = 2*self.ground_width-new_y
        
        return new_x, new_y

    def update_task(self, ue_id, computation_ratio):
        if self.task_list[ue_id] != 0:
            self.task_list[ue_id] = (1-computation_ratio)*self.task_list[ue_id]
        
        computation_done = self.t_com*(self.f_ue / self.s)
        
        for i in range(len(self.task_list)):
            if i != ue_id:
                self.task_list[i] = max(0, self.task_list[i]-computation_done)
        
        self.curr_sum_task_size = sum(self.task_list)
    
    def random_movement_ue(self, delay):
        for i in range(self.M):                                                 
            tmp = np.random.rand(2)
            theta_ue = tmp[0] * np.pi * 2
            dis_ue = tmp[1] * delay * self.v_ue
            self.loc_ue_list[i][0] = self.loc_ue_list[i][0] + math.cos(theta_ue) * dis_ue
            self.loc_ue_list[i][1] = self.loc_ue_list[i][1] + math.sin(theta_ue) * dis_ue
            self.loc_ue_list[i] = np.clip(self.loc_ue_list[i], 0, self.ground_width)

    
    def com_delay(self, offloading_ratio, ue_id):
        dx = self.loc_uav[0] - self.loc_ue_list[ue_id][0]
        dy = self.loc_uav[1] - self.loc_ue_list[ue_id][1]
        dh = self.height
        
        dist_uav_ue = np.sqrt(dx * dx + dy * dy + dh * dh)
        p_noise = self.p_noisy_los
        
        g_uav_ue = abs(self.alpha0 / dist_uav_ue ** 2)
        trans_rate = self.B * math.log2(1 + self.p_uplink * g_uav_ue / p_noise)
        
        task_size = self.task_list[ue_id]
        
        t_tr = offloading_ratio * task_size / trans_rate
        t_edge_com = offloading_ratio * task_size / (self.f_uav / self.s)
        t_local_com = (1 - offloading_ratio) * task_size / (self.f_ue / self.s)
        if t_tr < 0 or t_edge_com < 0 or t_local_com < 0:
            raise Exception(print("+++++++++++++++++!! error !!+++++++++++++++++++++++"))

        return max([t_tr/10 + t_edge_com, t_local_com])

    def step(self, action):
        
        action = (action+1)/2
        if action[0] == 1:
            ue_id = self.M - 1
        else:
            ue_id = int(self.M * action[0])
        theta = action[1] * np.pi * 2
        speed = action[2] * self.flight_speed
        offloading_ratio = action[3]
        print('UE ID:', ue_id, 'Speed:', speed, 'Angle:', theta, 'Ratio:', offloading_ratio)
        
        b = False
        if self.task_list[ue_id] == 0:
            b = True
        
        task_size = self.task_list[ue_id]

        dist_fly = speed * self.t_fly
        e_fly = (speed) ** 2 * self.m_uav * self.t_fly * 0.5

        dx = dist_fly * math.cos(theta)
        dy = dist_fly * math.sin(theta)
        loc_uav_new_x, loc_uav_new_y = self.getUENewLoc(x=self.loc_uav[0], y=self.loc_uav[1], dx=dx, dy=dy)
        self.loc_uav[0] = loc_uav_new_x
        self.loc_uav[1] = loc_uav_new_y
        
        computation_ratio = 0
        if b == False:
            act_delay = self.com_delay(offloading_ratio, ue_id)
            delay = min(self.t_com, act_delay)
            computation_ratio = delay/act_delay

        t_server = computation_ratio * offloading_ratio * task_size / (self.f_uav / self.s)
        e_server = self.r * self.f_uav ** 3 * t_server

        init_sum = self.curr_sum_task_size
        init_zero = self.task_list.tolist().count(0)

        copy_task_list = np.copy(self.task_list)
        
        self.update_task(ue_id, computation_ratio)

        for i in range(self.M):
            if copy_task_list[i] != 0 and self.task_list[i] == 0:
                self.end_time[i] = self.step_
        
        final_sum = self.curr_sum_task_size
        final_zero = self.task_list.tolist().count(0)
        
        reward = 0
        reward += (init_sum - final_sum)/1000000
        reward += (final_zero - init_zero)*5
                                  
        # self.e_battery_uav = self.e_battery_uav - e_fly - e_server
        self.random_movement_ue(self.t_fly)
        self.update_state()
        
        if b == True:
            reward -= 50
        
        print('Reward', reward)

        return self.state, reward, (self.curr_sum_task_size == 0), b