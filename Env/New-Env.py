import numpy as np
import math

class UAVEnv(object):
  
    height = ground_length = ground_width = 100
    SUM_TASK_SIZE = curr_sum_task_size = 300 * 1048576    # Total computing tasks 500 Mbits
    loc_uav = [50, 50]
    bandwidth_nums = 1
    B = bandwidth_nums * 10 ** 6                          # Bandwidth 1MHz
    p_noisy_los = 10 ** (-13)                             # Noise power -100dBm
    p_noisy_nlos = 10 ** (-11)                            # Noise power -80dBm
    flight_speed = 50.                                    # Flight speed 50m/s
    f_ue = 6e8                                            # UE computing frequency : 0.6GHz
    f_uav = 6e9                                           # UAV computing frequency : 1.2GHz
    r = 10 ** (-27)        
    s = 1000                                              # number of cpu cycles required for unit bit processing is 1000
    p_uplink = 0.1                                        # Uplink transmission power 0.1W
    alpha0 = 1e-5                                         # Reference channel gain at 1m distance -30dB = 0.001, -50dB = 1e-5
    v_ue = 1                                              # ue moving speed 1m/s
    t_fly = 1
    t_com = 5
    m_uav = 9.65                                          # uav mass/kg
    e_battery_uav = 50000000                              # uav battery power: 500kJ. 

    ##### ues #####
    M = 4                                                 # Number of UEs
#     block_flag_list = np.random.randint(0, 2, M)  
    loc_ue_list = np.random.randint(0, 101, size=[M, 2])  # Location information: x is random between 0-100
    task_list = np.random.dirichlet(np.ones(M))*SUM_TASK_SIZE

    action_bound = [-1, 1]                                # Corresponding to the tahn activation function
    action_dim = 4                                        # UE id, flight angle, flight speed, offloading ratio
    #state_dim = 4 + M * 4                                 # uav battery, uav loc, sum task size, ue loc, ue task size, ue block_flag
    state_dim = 4 + M * 3
    
    end_time = np.zeros(M)

    def __init__(self):
        self.update_state()

    def reset_uav(self):
        # print('Resetting UAV')
        self.e_battery_uav = 50000000
        self.loc_uav = [50, 50]
        self.update_state()

    def reset_ue(self):
        # print('Resetting UE')
        self.end_time = np.zeros(self.M)
        self.task_list = np.random.dirichlet(np.ones(self.M))*self.SUM_TASK_SIZE
        self.curr_sum_task_size = self.SUM_TASK_SIZE
        self.loc_ue_list = np.random.randint(0, 101, size=[self.M, 2])
#         self.block_flag_list = np.random.randint(0, 2, self.M)
        self.update_state()

    def print_state(self):
      print('Battery :', self.e_battery_uav, '\nUAV loc :', self.loc_uav, '\nTask :', self.curr_sum_task_size, self.task_list, '\nUE location :', np.ravel(self.loc_ue_list))#, '\nBlock Flag :', self.block_flag_list)

    def update_state(self):
        # print('Updating State')
        self.state = np.append(self.e_battery_uav, self.loc_uav)
        self.state = np.append(self.state, self.curr_sum_task_size)
        self.state = np.append(self.state, np.ravel(self.loc_ue_list))
        self.state = np.append(self.state, self.task_list)
#         self.state = np.append(self.state, self.block_flag_list)   
    
    def getUENewLoc(self, x, y, dx, dy):

        new_x = (x+dx+2*self.ground_length)%2*self.ground_length
        if new_x > self.ground_length:
            new_x = 2*self.ground_length-new_x
        
        new_y = (y+dy+2*self.ground_width)%2*self.ground_width
        if new_y > self.ground_width:
            new_y = 2*self.ground_width-new_y
        
        return new_x, new_y
    
    def get_offloading_ratio(self, ue_id):
        dx = self.loc_uav[0] - self.loc_ue_list[ue_id][0]
        dy = self.loc_uav[1] - self.loc_ue_list[ue_id][1]
        dh = self.height
        
        dist_uav_ue = np.sqrt(dx * dx + dy * dy + dh * dh)
        p_noise = self.p_noisy_los
        
#         if self.block_flag_list[ue_id] == 1:
#             p_noise = self.p_noisy_nlos
        
        g_uav_ue = abs(self.alpha0 / dist_uav_ue ** 2)
        trans_rate = self.B * math.log2(1 + self.p_uplink * g_uav_ue / p_noise)
        offloading_ratio = (self.s/self.f_ue)/((self.s/self.f_ue) + (self.s/self.f_uav) + (1/trans_rate))
        
        return offloading_ratio
    
    def com_delay(self, offloading_ratio, ue_id):
        dx = self.loc_uav[0] - self.loc_ue_list[ue_id][0]
        dy = self.loc_uav[1] - self.loc_ue_list[ue_id][1]
        dh = self.height
        
        dist_uav_ue = np.sqrt(dx * dx + dy * dy + dh * dh)
        p_noise = self.p_noisy_los
        
#         if self.block_flag_list[ue_id] == 1:
#             p_noise = self.p_noisy_nlos
        
        g_uav_ue = abs(self.alpha0 / dist_uav_ue ** 2)                            # channel gain
        trans_rate = self.B * math.log2(1 + self.p_uplink * g_uav_ue / p_noise)   # Uplink transfer rate bps
        
        task_size = self.task_list[ue_id]
        
        t_tr = offloading_ratio * task_size / trans_rate                          # Upload delay
        t_edge_com = offloading_ratio * task_size / (self.f_uav / self.s)         # Latency calculation on UAV edge server
        t_local_com = (1 - offloading_ratio) * task_size / (self.f_ue / self.s)   # local computing delay
        if t_tr < 0 or t_edge_com < 0 or t_local_com < 0:
            raise Exception(print("+++++++++++++++++!! error !!+++++++++++++++++++++++"))
#         print('--', t_tr, '--', t_edge_com)
        return max([t_tr/10 + t_edge_com, t_local_com])                              # flight time impact factor
    
    def update_task(self, ue_id, computation_ratio):
        if self.task_list[ue_id] != 0:
            self.task_list[ue_id] = (1-computation_ratio)*self.task_list[ue_id]
        
        computation_done = self.t_com*(self.f_ue / self.s)
        
        for i in range(len(self.task_list)):
            if i != ue_id:
                self.task_list[i] = max(0, self.task_list[i]-computation_done)
        
        self.curr_sum_task_size = sum(self.task_list)
        
    def random_movement_ue(self, delay):
        # print('UE Random Movement')
        
        for i in range(self.M):                                                 
            tmp = np.random.rand(2)
            theta_ue = tmp[0] * np.pi * 2                                       # ue random angle
            dis_ue = tmp[1] * delay * self.v_ue                                 # ue random distance
            self.loc_ue_list[i][0] = self.loc_ue_list[i][0] + math.cos(theta_ue) * dis_ue
            self.loc_ue_list[i][1] = self.loc_ue_list[i][1] + math.sin(theta_ue) * dis_ue
            self.loc_ue_list[i] = np.clip(self.loc_ue_list[i], 0, self.ground_width)

    def step(self, action, step):     # 0: UE id, 1: flight angle, 2: flight speed, XX 3: off-loading ratio
        
        action = (action+1)/2
        if action[0] == 1:
            ue_id = self.M - 1
        else:
            ue_id = int(self.M * action[0])
        theta = action[1] * np.pi * 2
        speed = action[2] * self.flight_speed
        # offloading_ratio = self.get_offloading_ratio(ue_id)
        offloading_ratio = action[3]

#         print('UE ID:', ue_id, 'Speed:', speed, 'Angle:', theta, 'Ratio:', offloading_ratio)
        
        b = False
        if self.task_list[ue_id] == 0:
            b = True
        
        task_size = self.task_list[ue_id]
#         block_flag = self.block_flag_list[ue_id]

        # flight distance
        dist_fly = speed * self.t_fly                                             # 1s flight distance
        e_fly = (speed) ** 2 * self.m_uav * self.t_fly * 0.5                      # flight energy

        # Location of UAV after flight
        dx = dist_fly * math.cos(theta)
        dy = dist_fly * math.sin(theta)
        loc_uav_new_x, loc_uav_new_y = self.getUENewLoc(x=self.loc_uav[0], y=self.loc_uav[1], dx=dx, dy=dy)
        self.loc_uav[0] = loc_uav_new_x                                           # UAV location update
        self.loc_uav[1] = loc_uav_new_y
        
        computation_ratio = 0
        if b == False:
            act_delay = self.com_delay(offloading_ratio, ue_id)                       # Computation Delay
            delay = min(self.t_com, act_delay)
            computation_ratio = delay/act_delay

        # Server computing power consumption
        t_server = computation_ratio * offloading_ratio * task_size / (self.f_uav / self.s) # Latency Calculation on UAV
        e_server = self.r * self.f_uav ** 3 * t_server                                      # Computation Energy on UAV

        init_sum = self.curr_sum_task_size
        init_zero = self.task_list.tolist().count(0)

        copy_task_list = np.copy(self.task_list)
        
        self.update_task(ue_id, computation_ratio)

        for i in range(self.M):
            if copy_task_list[i] != 0 and self.task_list[i] == 0:
                self.end_time[i] = step
        
        final_sum = self.curr_sum_task_size
        final_zero = self.task_list.tolist().count(0)
        
        reward = 0
        reward += (init_sum - final_sum)/1000000
        reward += (final_zero - init_zero)*5


        # block_flag_list = np.random.randint(0, 2, self.M)                         # random block after movement                                         
        # self.e_battery_uav = self.e_battery_uav - e_fly - e_server                # Battery remaining 
        self.random_movement_ue(self.t_fly)
        self.update_state()
        
        if b == True:
            reward -= 50
        return self.state, reward, (self.curr_sum_task_size == 0), b
            
