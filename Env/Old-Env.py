import  math
import  random

import  numpy  as  np


class  UAVEnv ( object ):
    height  =  ground_length  =  ground_width  =  100   # The length and width of the field are both 100m, and the flight height of UAV is also
    sum_task_size  =  100  *  1048576   # total calculation task 60 Mbits --> 60 80 100 120 140
    loc_uav  = [ 50 , 50 ]
    bandwidth_nums  =  1
    B  =  bandwidth_nums  *  10  **  6   # bandwidth 1MHz
    p_noisy_los  =  10  ** ( - 13 )   # noise power -100dBm
    p_noisy_nlos  =  10  ** ( - 11 )   # noise power -80dBm
    flight_speed  =  50.   # flight speed 50m/s
    # f_ue = 6e8 # The calculation frequency of UE is 0.6GHz
    f_ue  =  2e8   # The calculation frequency of UE is 0.6GHz
    f_uav  =  1.2e9   # The calculation frequency of UAV is 1.2GHz
    r  =  10  ** ( - 27 )   # Influence factor of chip structure on cpu processing
    s  =  1000   # The number of cpu circles required for unit bit processing is 1000
    p_uplink  =  0.1   # Uplink transmission power 0.1W
    alpha0  =  1e-5   # Reference channel gain when the distance is 1m -30dB = 0.001, -50dB = 1e-5
    T  =  320   # period 320s
    t_fly  =  1
    t_com  =  7
    delta_t  =  t_fly  +  t_com   # 1s flight, the last 7s is used for hover calculation
    v_ue  =  1     # ue moving speed 1m/s
    slot_num  =  int ( T  /  delta_t )   # 40 slots
    m_uav  =  9.65   #uav mass/kg
    e_battery_uav  =  500000   # uav battery power: 500kJ. ref: Mobile Edge Computing via a UAV-Mounted Cloudlet: Optimization of Bit Allocation and Path Planning

    ##################### ues ######################
    M  =  4   # Number of UEs
    block_flag_list  =  np . random . randint ( 0 , 2 , M )   # 4 ue, the occlusion of ue
    loc_ue_list  =  np . random . randint ( 0 , 101 , size = [ M , 2 ])   # position information: x is random in 0-100
    # task_list = np.random.randint(1572864, 2097153, M) # Random calculation task 1.5~2Mbits -> corresponding total task size 60
    task_list  =  np . random . randint ( 2097153 , 2621440 , M )   # random computing task 2~2.5Mbits -> 80
    # ue position transition probability
    # 0: The position remains unchanged; 1: x+1,y; 2:x,y+1; 3:x-1,y; 4:x,y-1
    # loc_ue_trans_pro = np.array([[.6, .1, .1, .1, .1],
    # [.6, .1, .1, .1, .1],
    # [.6, .1, .1, .1, .1],
    # [.6, .1, .1, .1, .1]])

    action_bound  = [ - 1 , 1 ]   # corresponds to tahn activation function
    action_dim  =  4   # The first digit represents the ue id of the service; the middle two digits represent the flight angle and distance; the last digit represents the unloading rate currently serving UE
    state_dim  =  4  +  M  *  4   # uav battery remain, uav loc, remaining sum task size, all ue loc, all ue task size, all ue block_flag

    def  __init__ ( self ):
        # uav battery remain, uav loc, remaining sum task size, all ue loc, all ue task size, all ue block_flag
        self . start_state  =  np . append ( self . e_battery_uav , self . loc_uav )
        self . start_state  =  np . append ( self . start_state , self . sum_task_size )
        self . start_state  =  np . append ( self . start_state , np . ravel ( self . loc_ue_list ))
        self . start_state  =  np . append ( self . start_state , self . task_list )
        self . start_state  =  np . append ( self . start_state , self . block_flag_list )
        self.state = self.start_state
    def  reset_env ( self ):
        self . sum_task_size  =  100  *  1048576   # total calculation task 60 Mbits -> 60 80 100 120 140
        self . e_battery_uav  =  500000   # uav battery power: 500kJ
        self.loc_uav = [ 50 , 50 ]
        self . loc_ue_list  =  np . random . randint ( 0 , 101 , size = [ self . M , 2 ])   # position information: x is random between 0-100
        self.reset_step ( )

    def  reset_step ( self ):
        # self.task_list = np.random.randint(1572864, 2097153, self.M) # Random calculation task 1.5~2Mbits -> 1.5~2 2~2.5 2.5~3 3~3.5 3.5~4
        # self.task_list = np.random.randint(2097152, 2621441, self.M) # Random calculation task 1.5~2Mbits -> 1.5~2 2~2.5 2.5~3 3~3.5 3.5~4
        # self.task_list = np.random.randint(2621440, 3145729, self.M) # Random calculation task 1.5~2Mbits -> 1.5~2 2~2.5 2.5~3 3~3.5 3.5~4
        self . task_list  =  np . random . randint ( 2621440 , 3145729 , self . M )   # Random computing task 1.5~2Mbits -> 1.5~2 2~2.5 2.5~3 3~3.5 3.5~4
        # self.task_list = np.random.randint(3145728, 3670017, self.M) # Random calculation task 1.5~2Mbits -> 1.5~2 2~2.5 2.5~3 3~3.5 3.5~4
        # self.task_list = np.random.randint(3670016, 4194305, self.M) # Random calculation task 1.5~2Mbits -> 1.5~2 2~2.5 2.5~3 3~3.5 3.5~4
        self . block_flag_list  =  np . random . randint ( 0 , 2 , self . M )   # 4 ue, the occlusion of ue

    def  reset ( self ):
        self.reset_env ( )
        # uav battery remain, uav loc, remaining sum task size, all ue loc, all ue task size, all ue block_flag
        self.state = np.append ( self.e_battery_uav , self.loc_uav )
        self.state = np.append ( self.state , self.sum_task_size )
        self . state  =  np . append ( self . state , np . ravel ( self . loc_ue_list ))
        self.state = np.append ( self.state , self.task_list )
        self.state = np.append ( self.state , self.block_flag_list )
        return  self . _get_obs ()

    def  _get_obs ( self ):
        # uav battery remain, uav loc, remaining sum task size, all ue loc, all ue task size, all ue block_flag
        self.state = np.append ( self.e_battery_uav , self.loc_uav )
        self.state = np.append ( self.state , self.sum_task_size )
        self . state  =  np . append ( self . state , np . ravel ( self . loc_ue_list ))
        self.state = np.append ( self.state , self.task_list )
        self.state = np.append ( self.state , self.block_flag_list )
        return  self . state

    def  step ( self , action ):   # 0: select the ue number of the service; 1: direction theta; 2: distance d; 3: offloading ratio
        step_redo  =  False
        is_terminal  =  False
        offloading_ratio_change  =  False
        reset_dist  =  False
        action  = ( action  +  1 ) /  2   # will take the action of value range -1~1 -> the action of 0~1. Avoid training the actor network tanh function to always take the boundary 0 when the original action_bound is [0,1]
        #####################################################################################
        # Improve ddpg, add a layer to the output layer to output discrete actions (the implementation result is wrong)
        # Using the shortest distance algorithm, there is an error. If the shortest distance drone has been parked on the head (wrong)
        # Random polling: first generate a random number queue, remove the UE after the service is completed, and randomly generate again when the queue is empty (logic is wrong)
        # Control variables are mapped to the value range of each variable
        if  action [ 0 ] ==  1 :
            ue_id  =  self . M  -  1
        else :
            ue_id  =  int ( self . M  *  action [ 0 ])

        theta  =  action [ 1 ] *  np . pi  *  2   # angle
        offloading_ratio  =  action [ 3 ]   # ue offloading rate
        task_size  =  self.task_list [ ue_id ]
        block_flag  =  self . block_flag_list [ ue_id ]

        # flight distance
        dis_fly  =  action [ 2 ] *  self . flight_speed  *  self . t_fly   # 1s flight distance
        # Flight energy consumption
        e_fly  = ( dis_fly  /  self . t_fly ) **  2  *  self . m_uav  *  self . t_fly  *  0.5   # ref: Mobile Edge Computing via a UAV-Mounted Cloudlet: Optimization of Bit Allocation and Path Planning

        # Position of the UAV after flight
        dx_uav  =  dis_fly  *  math . cos ( theta )
        dy_uav  =  dis_fly  *  math.sin ( theta )
        loc_uav_after_fly_x  =  self .loc_uav [ 0 ] +  dx_uav
        loc_uav_after_fly_y  =  self .loc_uav [ 1 ] +  dy_uav

        # server computing energy consumption
        t_server  =  offloading_ratio  *  task_size  / ( self . f_uav  /  self . s )   # Calculate the delay on the UAV edge server
        e_server  =  self . r  *  self . f_uav  **  3  *  t_server   # Calculate energy consumption on UAV edge server

        if  self . sum_task_size  ==  0 :   # calculation tasks are all completed
            is_terminal  =  True
            reward  =  0
        elif  self . sum_task_size  -  self . task_list [ ue_id ] <  0 :   # The calculation task of the last step does not match the calculation task of ue
            self.task_list = np.ones  ( self.M ) * self.sum_task_size
            reward  =  0
            step_redo  =  True
        elif  loc_uav_after_fly_x  <  0  or  loc_uav_after_fly_x  >  self . ground_width  or  loc_uav_after_fly_y  <  0  or  loc_uav_after_fly_y  >  self . ground_length :   # uav position is wrong
            # If it exceeds the boundary, the flight distance dist is set to zero
            reset_dist  =  True
            delay  =  self . com_delay ( self . loc_ue_list [ ue_id ], self . loc_uav , offloading_ratio , task_size , block_flag )   # calculate delay
            reward  =  -delay
            # Update the status of the next moment
            self . e_battery_uav  =  self . e_battery_uav  -  e_server   # remaining power of uav
            self . reset2 ( delay , self . loc_uav [ 0 ], self . loc_uav [ 1 ], offloading_ratio , task_size , ue_id )
        elif  self . e_battery_uav  <  e_fly  or  self . e_battery_uav  -  e_fly  <  e_server :   # uav power cannot support calculation
            delay  =  self . com_delay ( self . loc_ue_list [ ue_id ], np . array ([ loc_uav_after_fly_x , loc_uav_after_fly_y ]),
                                   0 , task_size , block_flag )   # calculate delay
            reward  =  -delay
            self . reset2 ( delay , loc_uav_after_fly_x , loc_uav_after_fly_y , 0 , task_size , ue_id )
            offloading_ratio_change  =  True
        else :   # The power supports flight, and the calculation task is reasonable, and the calculation task can be calculated within the remaining power
            delay  =  self . com_delay ( self . loc_ue_list [ ue_id ], np . array ([ loc_uav_after_fly_x , loc_uav_after_fly_y ]),
                                   offloading_ratio , task_size , block_flag )   # calculate delay
            reward  =  -delay
            # Update the status of the next moment
            self . e_battery_uav  =  self . e_battery_uav  -  e_fly  -  e_server   # remaining power of uav
            self .loc_uav [ 0 ] = loc_uav_after_fly_x # position of uav after flight   
            self.loc_uav [ 1 ] = loc_uav_after_fly_y
            self . reset2 ( delay , loc_uav_after_fly_x , loc_uav_after_fly_y , offloading_ratio , task_size ,
                                           ue_id )    # Reset ue task size, remaining total task size, ue position, and record to file

        return  self . _get_obs (), reward , is_terminal , step_redo , offloading_ratio_change , reset_dist

    # Reset ue task size, remaining total task size, ue position, and record to file
    def  reset2 ( self , delay , x , y , offloading_ratio , task_size , ue_id ):
        self . sum_task_size  -=  self . task_list [ ue_id ]   # remaining tasks
        for  i  in  range ( self . M ):   # ue position after random movement
            tmp  =  np.random.rand ( 2 )
            theta_ue  =  tmp [ 0 ] *  np . pi  *  2   # ue random movement angle
            dis_ue  =  tmp [ 1 ] *  self . delta_t  *  self . v_ue   # ue random movement distance
            self .loc_ue_list [ i ] [ 0 ] =  self .loc_ue_list [ i ] [ 0 ] + math .cos ( theta_ue ) * dis_ue  
            self .loc_ue_list [ i ] [ 1 ] =  self .loc_ue_list [ i ] [ 1 ] + math .sin ( theta_ue ) * dis_ue
            self.loc_ue_list [ i ] = np.clip ( self.loc_ue_list [ i ] , 0 , self.ground_width )
        self . reset_step ()   # ue random calculation task 1~2Mbits # 4 ue, the occlusion of ue
        # Record UE cost
        file_name  =  'output.txt'
        # file_name = 'output_ddpg_' + str(self.bandwidth_nums) + 'MHz.txt'
        with  open ( file_name , 'a' ) as  file_obj :
            file_obj .write ( " \ n UE-"  +  '{:d}' .format ( ue_id ) + " , task size: " + '{:d}' .format ( int ( task_size ) ) + ", offloading ratio: " + '{:.2f}' . format ( offloading_ratio ))      
            file_obj . write ( " \n delay:"  +  '{:.2f}' . format ( delay ))
            file_obj .write ( " \  n UAV hover loc:" +  "["  +  '{:.2f}' .format ( x ) + '  , '  +  '{:.2f}' .format ( y ) + '  ]' )   # output retains two results


    # calculate cost
    def  com_delay ( self , loc_ue , loc_uav , offloading_ratio , task_size , block_flag ):
        dx  =  loc_uav [ 0 ] -  loc_ue [ 0 ]
        dy  =  loc_uav [ 1 ] -  loc_ue [ 1 ]
        dh  =  self . height
        dist_uav_ue  =  np.sqrt ( dx * dx + dy * dy + dh * dh )          
        p_noise  =  self.p_noisy_los
        if  block_flag  ==  1 :
            p_noise  =  self.p_noisy_nlos
        g_uav_ue  =  abs ( self . alpha0  /  dist_uav_ue  **  2 )   # channel gain
        trans_rate  =  self . B  *  math . log2 ( 1  +  self . p_uplink  *  g_uav_ue  /  p_noise )   # Uplink transmission rate bps
        t_tr  =  offloading_ratio  *  task_size  /  trans_rate   # upload delay, 1B=8bit
        t_edge_com  =  offloading_ratio  *  task_size  / ( self . f_uav  /  self . s )   # Calculate the delay on the UAV edge server
        t_local_com  = ( 1  -  offloading_ratio ) *  task_size  / ( self . f_ue  /  self . s )   # Local calculation delay
        if  t_tr  <  0  or  t_edge_com  <  0  or  t_local_com  <  0 :
            raise  Exception ( print ( "++++++++++++++++++!! error !!++++++++++++++++++++++++ +" ))
        return  max ([ t_tr  +  t_edge_com , t_local_com ])   # flight time impact factor