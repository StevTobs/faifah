from copy import deepcopy
import numpy as np
import pandas as pd
import cmath
import math
import datetime
import pypsa
from numpy import genfromtxt
from colorama import Fore, Back, Style
from termcolor import colored

# from <FOLDER NAME> import <NAME of .py>
from faifah import ieee_test_system
# import ieee_test_system

# Require :
# - numpy
# - pandas

# - linopy 
# - cartopy

class Grid:

    def __init__( self, nbus, NETWORK_ = '', LINE_DATA_CSV='', BUS_DATA_CSV='', MVA_base = np.nan , KV_base = np.nan, SLACK_POS = np.nan ):

        if len(LINE_DATA_CSV) != 0 and len(BUS_DATA_CSV) != 0  :
            
            self.load_data,  self.line_data, self.nBus = self.query_csv( nbus, LINE_DATA_CSV, BUS_DATA_CSV)
            
        elif NETWORK_ == 'IEEE5' :
            print('IEEE5-bus test distribution system')
            self.load_data,  self.line_data, self.nBus = ieee_test_system.load_IEEE5()
        
        elif NETWORK_ == 'IEEE33' :
            print('IEEE33-bus test distribution system')
            self.load_data,  self.line_data, self.nBus = ieee_test_system.load_IEEE33()
        
        elif NETWORK_ == 'IEEE69' :
            print('IEEE69-bus test distribution system')
            self.load_data,  self.line_data, self.nBus = ieee_test_system.load_IEEE69()

        else :
            print('Invalid')
        
        self.slack_bus_pos  = SLACK_POS 
        self.MVA_base       = MVA_base   
        self.KV_base        = KV_base
        self.Z_base         = KV_base * KV_base / MVA_base 
        self.Y_base         = 1 / self.Z_base


        self.total_kW_demand    = self.load_data.P_KW.sum() 
        self.total_kW_supply    = 0
        self.total_kVAr_demand  = self.load_data.Q_KVAr.sum()
        self.total_kVAr_supply  = 0


        self.total_kW_loss = 0
        self.total_kVAr_loss = 0

        self.transmission_kW_loss_LPP =0
        self.transmission_kW_loss_LQQ =0
        self.transmission_kW_loss_LPQ =0

        # My special parameters (Transfer Voltage Stability Index)
            
        self.S = np.zeros(nbus, dtype=complex)
        self.V = np.zeros(nbus, dtype=complex)

        self.Psi_k = np.zeros(nbus, dtype=complex)
        self.Psi_p = np.zeros(nbus)
        self.Psi_q = np.zeros(nbus)
        
        self.Psi_P_SNB = np.zeros(nbus)
        self.Psi_Q_SNB = np.zeros(nbus)

        self.Ybus= np.zeros((nbus,nbus), dtype=complex) 

        self.Psi_P_SNB = np.zeros(nbus)
        self.Psi_Q_SNB = np.zeros(nbus)

        # Will be removed
        self.Theta = np.zeros(nbus)  
        self.Psi_d = np.zeros(nbus, dtype=complex)
        self.Psi_dk = np.zeros(nbus, dtype=complex)
        self.alpha = np.zeros(nbus)
        self.reset_PYPSAnetwork()
        self.LoadFlow()

        self.org_network  = deepcopy(self.network)
 
    

    
    def query_csv(self, nbus, LINE_DATA_CSV, BUS_DATA_CSV):

        print("Query from the external csv files")
        # self.LINE_DATA_CSV = LINE_DATA_CSV 
        # self.BUS_DATA_CSV = BUS_DATA_CSV 

        #Line data 
        #Cleaning data from CSV
        dummy_data = genfromtxt(LINE_DATA_CSV, delimiter=',')
        dummy_data[:,0] = np.arange(1,len(dummy_data)+1).astype(int)
        dummy_data[:,1] = dummy_data[:,1].astype(int)
        dummy_data[:,2] = dummy_data[:,2].astype(int)
        
        line_data = pd.DataFrame( dummy_data [:,:5] , columns=['No','FromBus','ToBus','R','X'] )
        # line_data = pd.read_csv( LINE_DATA_CSV  , names=['No','FormBus','ToBus','R','X','S'] ).dropna(axis=1)
        line_data.set_index('No',inplace=True)
        line_data['FromBus'] = line_data['FromBus'].astype(np.uint8)
        line_data['ToBus'] = line_data['ToBus'].astype(np.uint8)
        #Convert to numeric, good guide(https://stackoverflow.com/questions/15891038/change-column-type-in-pandas)
        line_data.R = pd.to_numeric(line_data.R).astype(float).to_frame()
        line_data.X = pd.to_numeric(line_data.X).astype(float).to_frame()
        del dummy_data 
        # print(self.line_data)

        #Load data
        #Cleaning data from CSV
        dummy_data = genfromtxt( BUS_DATA_CSV, delimiter=',')
        dummy_data[:,0] = np.arange(1,nbus+1).astype(int)
        dummy_data  = dummy_data [:,:3]
        load_data = pd.DataFrame( dummy_data , columns=['BusID','P_KW','Q_KVAr'] )
        load_data['BusID'] = load_data['BusID'].astype(np.uint8)
        del dummy_data 
        # print(load_data)

        #Convert to numeric
        load_data.P_KW = pd.to_numeric(load_data.P_KW).astype(float)
        load_data.Q_KVAr= pd.to_numeric(load_data.Q_KVAr).astype(float)
        

        return load_data,  line_data, nbus

    def reset_PYPSAnetwork(self):

        self.network = pypsa.Network()

        self.Bus = pd.DataFrame( np.zeros((self.nBus,4)), columns=['v_mag','v_ang','P_pu','Q_pu']  )

        #Add busess           
        for i in range(self.nBus):
            self.network.add("Bus",name="Bus No {}".format(int(i+1)), v_mag_pu_set = 1.0)

        #Add loads           
        for i in range(self.nBus):
            # try :
            self.network.add("Load", "Load No {}".format(int(i+1)),
                bus = "Bus No {}".format(int(i+1)),
                p_set =list(  self.load_data['P_KW'][self.load_data['BusID']==i+1] / 1000 / self.MVA_base)[0],
                q_set = list( self.load_data['Q_KVAr'][self.load_data['BusID']==i+1] / 1000 / self.MVA_base)[0])
            # except :
            #   self.network.add("Load", "Load No {}".format(i+1),
            #       bus = "Bus No {}".format(i+1),
            #       p_set = 0,
            #       q_set = 0)

        #Add lines
        for i in range(len(self.line_data.ToBus.loc[:])):
            self.network.add("Line","Line {}".format(str(int(self.line_data.FromBus.iloc[i]))+str('-')+ str(int(self.line_data.ToBus.iloc[i]))),              
                        bus0 = "Bus No {}".format(int(self.line_data.FromBus.iloc[i])),
                        bus1 = "Bus No {}".format(int(self.line_data.ToBus.iloc[i])),
                        r = self.line_data.R.iloc[i] / self.Z_base,
                        x = self.line_data.X.iloc[i] / self.Z_base)

        # print(network.lines.head())
        # print('-----------------------------------------------')
        #slack generator
        self.network.add("Generator","Slack Gen",
                bus = "Bus No {}".format(int(self.slack_bus_pos)),
                control = 'Slack',
                v_mag_pu_set = 1.00 )


    def LoadFlow(self): 

        # slack_id = self.slack_bus_pos - 1   
        self.network.pf()

        self.Bus.v_mag = np.array( self.network.buses_t.v_mag_pu.iloc[0][:] )
        self.Bus.v_ang = np.array( self.network.buses_t.v_ang.iloc[0][:] )
        self.Bus.P_pu = np.array( self.network.buses_t.p.iloc[0][:] )
        self.Bus.Q_pu = np.array( self.network.buses_t.q.iloc[0][:] )

        sum_supply_p = self.network.generators_t.p.sum().sum()
        sum_demand_p = self.network.loads_t.p.sum().sum()
        sum_supply_q = self.network.generators_t.q.sum().sum()
        sum_demand_q = self.network.loads_t.q.sum().sum()

        self.total_kW_supply    = sum_supply_p* self.MVA_base*1000
        self.total_kVAr_supply  =  sum_supply_q* self.MVA_base*1000

        self.total_kW_demand   = sum_demand_p* self.MVA_base*1000
        self.total_kVAr_demand   = sum_demand_q * self.MVA_base *1000

        self.total_kW_loss     = (sum_supply_p - sum_demand_p)* self.MVA_base*1000
        self.total_kVAr_loss   = (sum_supply_q - sum_demand_q)* self.MVA_base*1000


        self.Ybus =  np.copy(self.GetYbus())

        # Update my special parameters (Transfer Voltage Stability Index)
        Y_bus   =  np.copy(self.GetYbus())

        for i in range(self.nBus ):
            Ykk_ph   = Y_bus[i,i]
            Ykk   = abs(Y_bus[i,i])
            # Ykk_ph  = complex(  np.real( Y_bus[i,i]), np.imag(Y_bus[i,i]))
            # Ykk_ph.real =  Y_bus[i,i].real
            # Ykk_ph.imag =  Y_bus[i,i].imag 
            theta_kk = np.angle( Y_bus[i,i])

            Pk = self.Bus.loc[i].P_pu 
            Qk = self.Bus.loc[i].Q_pu 
            Sk_ph = np.empty([], dtype=np.complex128)
            Sk_ph.real  = Pk


            Sk_ph.imag  = Qk   


            phi_k   =  np.angle(Sk_ph) 
            Y_bus   =  np.copy(self.GetYbus())


            self.V.real[i] = self.Bus.v_mag[i]* np.cos(phi_k)
            self.V.imag[i] = self.Bus.v_mag[i]* np.sin(phi_k)
            
            # self.Phi[i] = phi_k 
            self.Theta[i] = theta_kk 

            self.S.real[i]  = self.Bus.loc[i].P_pu 
            self.S.imag[i]  = self.Bus.loc[i].Q_pu 

            self.Psi_p[i] =  np.real(np.conj(self.S[i])  / Y_bus[i,i])

            self.Psi_q[i] = np.imag(np.conj(self.S[i])  / Y_bus[i,i])

            self.Psi_k[i] = ( (np.conj(Sk_ph) / Y_bus[i,i])  -  np.abs( self.V[i])**2 )  / np.conj(self.V[i])

            self.alpha[i] =  self.Psi_q[i]/self.Psi_p[i] if abs(self.Psi_p[i]) > 10^-116 else np.nan


            # Will be removed
            self.Psi_d[i] = (np.conj(Sk_ph)  / Y_bus[i,i] ) -  self.Bus.v_mag[i]**2 
            self.Psi_dk[i] = self.Psi_d[i] / np.conj(self.V[i])


    def addGen(self, BusID, P_KW, Q_KVAr, Control, Gen_name):

        #Example
        # BusID = 2
        # P_KW = 40
        # Q_KVAr = 30
        # Control='PQ'
        # Gen_name = "Gen No 1"
        # IEEE5.addGen(BusID, P_KW, Q_KVAr, Control, Gen_name)# BusID = 2
        # P_KW = 40
        # Q_KVAr = 30
        # Control='PQ'
        # IEEE5.addGen(BusID, P_KW, Q_KVAr, Control)

        # Control (https://pypsa.readthedocs.io/en/latest/components.html#generator)
        # P,Q,V control strategy for PF, must be “PQ”, “PV” or “Slack”.

        self.network.add("Generator", Gen_name, 
                    bus = "Bus No {}".format(BusID), # BusID: Name of bus to which generator is attached
                p_set = float(P_KW)/self.MVA_base/1000,
                q_set = float(Q_KVAr)/self.MVA_base/1000,
                control = Control)   # Control: P,Q,V control strategy for PF,  must be “PQ”, “PV” or “Slack” 
        self.LoadFlow()
        # slack_id = self.slack_bus_pos - 1 

    def Report(self):

        # self.VSI()['VSI'][1:].mean() #Exclude Slack BUS

        # print( 'Active Power Suppy (From Grid) (kW): ', self.Grid_kW_supply)
        # print( 'Reactive Power Suppy (From Grid) (kVAr): ',self.Grid_kVAr_supply )
        print( 'Total Active Power Suppy (kW): ', self.total_kW_supply )
        print( 'Total Reactive Power Suppy (kVAr): ',self.total_kVAr_supply  )
        print( '-------------------------------------------')
        # print( 'Total (Reviece) Power Condumption (MW): ',sum(- IEEE33.Bus.P_pu.iloc[2:] ) * IEEE33.MVA_base)
        print( 'Total Power Demand (kW): ',self.total_kW_demand )
        # print( 'Total (Reviece) Power Condumption (MW): ',sum(- IEEE33.Bus.P_pu.iloc[2:] ) * IEEE33.MVA_base)
        print( 'Total Reactive Power Demand (KVAr): ', self.total_kVAr_demand )
        print( '-------------------------------------------')

        print( 'Total Active Power Loss (kW): ', self.total_kW_loss   )
        print( 'Total Reactive Power Loss (kVar): ', self.total_kVAr_loss  )

        # print( 'Mean Voltage Stability Index (MVSI): ', self.VSI()['VSI'][1:].mean() , "(Exclude the Slack bus's voltage)" )

        print( '---------------REPORT Transfer Voltage Stability Parameters------------------------')
        Y_bus   =  np.copy(self.GetYbus())
        print( '|Bus No.  |        v        |   p   |   q   |  phi  |    Ykk   |theta_k| abs(Psi_P)| abs(Psi_Q) |   Psi   | alpha | PF |') 
        for i in range(self.nBus):
            Ykk_ph  = complex(  np.real( Y_bus[i,i]), np.imag(Y_bus[i,i]))
            Pk = self.Bus.loc[i].P_pu 
            Qk = self.Bus.loc[i].Q_pu 
            Sk_ph   =  complex( Pk , Qk  )
            # Sk_ph =  cmath.polar(Sk_ph)[1]
            phi_k   =  cmath.polar(Sk_ph)[1]
            theta_kk = cmath.polar(Ykk_ph)[1]
            Y_bus   =  np.copy(self.GetYbus())
            Ykk     =  -cmath.polar(Ykk_ph)[0]
            theta_kk = cmath.polar(Ykk_ph)[1]
            Vk = np.abs( self.V[i] )
            delta_k = np.angle( self.V[i] )


            # # print('Ykk :', Ykk)
            # self.Psi_p[i] = Pk* np.cos( phi_k + theta_kk )/ Ykk 
            # self.Psi_q[i] = -Qk* np.sin( phi_k + theta_kk )/ Ykk 
            # self.alpha[i] = self.Psi_q[i]/self.Psi_p[i]
            # self.Psi_d[i] = abs( (Sk_ph/Ykk_ph) - np.power( Vk, 2)) 

            # print()
            # print(f' Bus {i+1} {Pk:.03f}')
            print(f'|  Bus {1+i:d}  | {Vk:.03f} rad: {delta_k :.02f}| {Pk:.03f} | {Qk:.02f} | {phi_k:.02f} | {Ykk:.02f} | {theta_kk:.02f} | {self.Psi_p[i]:0.05f} | {self.Psi_q[i]:.05f} | {self.Psi_k[i]:.02f} | {self.alpha[i]:0.02f} |{np.cos(phi_k ):0.02f} |') 


        print( '------------------------------------------------------------------------------------------------------')

    def GetYbus(self):

        network = self.network
        nBus = self.nBus 

        Y_bus = np.zeros(( nBus, nBus), dtype=complex)

        for i in range(len( network.lines)):
            fromBus = int( network.lines['bus0'].iloc[i].split(' ')[2])
            toBus = int( network.lines['bus1'].iloc[i].split(' ')[2])

            # Y_bus[fromBus-1,toBus-1] = (((network.lines.r.iloc[i]+network.lines.x.iloc[i] *1j)*self.Z_base )**(-1))/self.Y_base
            # Y_bus[toBus-1,fromBus-1] = (((network.lines.r.iloc[i]+network.lines.x.iloc[i] *1j)*self.Z_base)**(-1))/self.Y_base
            Y_bus[fromBus-1,toBus-1] = -(((network.lines.r.iloc[i]+network.lines.x.iloc[i] *1j))**(-1))
            Y_bus[toBus-1,fromBus-1] = -(((network.lines.r.iloc[i]+network.lines.x.iloc[i] *1j))**(-1))

            if fromBus-1  != toBus-1 :
                Y_bus[fromBus-1,fromBus-1 ] = Y_bus[fromBus-1,fromBus-1 ] - Y_bus[fromBus-1,toBus-1] 
                Y_bus[toBus-1,toBus-1 ] = Y_bus[toBus-1,toBus-1 ] - Y_bus[fromBus-1,toBus-1]  



        return Y_bus
        
    def L_index(self, factor_load ):
        from copy import deepcopy
        Network_dummy = deepcopy(self)

        Y_bus = np.copy(self.GetYbus())
        L = np.zeros( self.nBus ,dtype=float)

        Network_dummy.network.loads.p_set.iloc[:] = Network_dummy.network.loads.p_set.iloc[:] * factor_load 
        Network_dummy.network.loads.q_set.iloc[:] = Network_dummy.network.loads.q_set.iloc[:] * factor_load       


        S_corr = np.zeros(self.nBus, dtype=complex)
        Y_plus = np.zeros(( self.nBus , self.nBus ), dtype=complex)
        Z = np.linalg.inv(self.Ybus)
        Z_conj = np.zeros(( self.nBus , self.nBus ), dtype=complex)
        S_plus = np.zeros(self.nBus, dtype=complex)
            
        for j in range(self.nBus):
            for i in range(self.nBus):

                # Z[j,j] = 1/self.Ybus[j,j]

                Z_conj[j,i] = np.conjugate(Z[j,i])
                Z_conj[j,j] = np.conjugate(Z[j,j])
                Y_plus[j,j] = Z[j,j]**-1
                # print( Z_conj[i,j])

            if i != j:
                S_corr[j] = S_corr[j] + self.V[j]* ( Z_conj[j,i] * self.S[i] / Z_conj[j,j] /self.V[i] ) 
                
            S_plus[j]  = self.S[j] + S_corr[j]

            
                
            L[j] = float("{:.5f}".format(abs( S_plus[j]/ ( np.conjugate( self.Ybus[j,j])) / ( (self.V[j]) ** 2)   )))


            # print('S_corr[j]',S_corr[j])
            # print('Y_plus[j,j]',Y_plus[j,j])
            # print('S_plus[j]',S_plus[j])
            # print('self.V[j]',self.V[j])
            # print("-------")
        return L

    def Find_one_weak_bus( self, load_factor, off_load , inplace ):

        privot = 1 #Exclude the slack bus
        if inplace:
            print( "inplace mode")
            maxL = np.max( self.L_index(  load_factor, off_load ) )
        else :
            print( "None inplace mode")
            maxL = np.max( self.L_index(  load_factor, []) )

        MaxL_keep =0
        # print(self.L_index(  load_factor, [])  )  
        print("Max{L} at none-disable loads : ", maxL  )
        # print(off_load)
        weakBus_ind = []


        while privot < self.nBus :

            if not privot + 1 in off_load:
                dis_buses = [privot+1] +  off_load  if inplace else [privot+1]
                dis_buses = dis_buses.remove([]) if [] in dis_buses else dis_buses

                maxL_dmm =  max( self.L_index(  load_factor, dis_buses ) ) 
                MaxL_curr = maxL - maxL_dmm

            if MaxL_keep  <=  MaxL_curr :
                MaxL_keep = MaxL_curr
                maxL_min= maxL_dmm 
                weakBus_ind = privot+1
                # print("Found!,",privot +1)

            privot = privot + 1

        print("At load factor:", load_factor,"The weakest bus : ",off_load + [ weakBus_ind ] , " max(L):",maxL_min ,"Max L reduction (%)", [100*MaxL_keep/maxL if maxL!=0 else 'inf'])
        return weakBus_ind
        #------------------------------------------------s-------------------------------------------------- 
    def P_feasibleUpper(self, BusN, a_k):

        if BusN <= self.nBus and BusN >1 :
            Y_bus = self.GetYbus() 
            Ykk_ph = - Y_bus[BusN-1][BusN-1] 
            Ykk = cmath.polar(Ykk_ph)[0]
            theta_kk = cmath.polar(Ykk_ph)[1]
            Sk_ph = complex( self.Bus.loc[BusN-1].P_pu , self.Bus.loc[BusN-1].Q_pu ) #pu
            phi_k = cmath.polar(Sk_ph)[1]
            Vk =  self.Bus.loc[BusN-1].v_mag #pu
            Psi_dk = abs( self.Psi_dk[BusN-1])

            return - Psi_dk**2 * Ykk * (np.sqrt(a_k**2 +1) -1)/2/(a_k**2)/np.cos(phi_k + theta_kk)

        elif BusN <= 1  :

            print("A slack bus")
        else :
            print("A number is out of range.")

    def PVcurve(self,BusN, a_k, n_samples):
        Y_bus   = np.copy(self.GetYbus())
        Ykk_ph  =  - Y_bus[BusN-1][BusN-1] 
        Ykk     =  cmath.polar(Ykk_ph)[0]
        theta_kk = cmath.polar(Ykk_ph)[1]
        Sk_ph   =  complex( self.Bus.loc[BusN-1].P_pu , self.Bus.loc[BusN-1].Q_pu ) #pu
        phi_k   =  cmath.polar(Sk_ph)[1]
        Vk      =  self.Bus.loc[BusN-1].v_mag #pu
        Psi_dk  =  abs( self.Psi_dk[BusN-1])

        vk_pos = np.zeros( n_samples) 
        vk_neg = np.zeros( n_samples) 

        p_max = self.P_feasibleUpper(BusN, a_k)

        p_k = np.linspace(0.0001 , p_max  , n_samples)

        for i in range(1,n_samples):
            term_a    =   np.sqrt( np.power(Psi_dk,4) -(4*np.power(p_k[i],2)*np.power(a_k,2)*np.power( np.cos(phi_k+theta_kk) , 2)/ np.power(Ykk,2))+ (( 4*p_k[i]*np.power(Psi_dk,2)*np.cos(phi_k+theta_kk))/Ykk) )
            v_pos_temp =  float(  np.power(2,-0.5 ) * np.sqrt(  np.power(Psi_dk,2) + (2*p_k[i]*np.cos(phi_k+theta_kk)/Ykk ) + term_a ) )
            v_neg_temp =  float(  np.power(2,-0.5 ) * np.sqrt(  np.power(Psi_dk,2) + (2*p_k[i]*np.cos(phi_k+theta_kk)/Ykk ) - term_a ) )

            
            vk_pos[i] = v_pos_temp if not(np.isnan(v_pos_temp))   else vk_pos[i-1]
            vk_neg[i] = v_neg_temp if not(np.isnan(v_neg_temp))   else vk_neg[i-1]

        return [p_k, vk_pos, vk_neg ]

    def q_req(self,V_reg, BusN):
        i = BusN-1
        # Ykk_ph  =  - Y_bus[BusN-1][] 
        # Ykk     =  cmath.polar(Ykk_ph)[0]
        # theta_kk = cmath.polar(Ykk_ph)[1]
        # Sk_ph   =  complex( IEEE5.Bus.loc[BusN-1].P_pu , IEEE5.Bus.loc[BusN-1].Q_pu ) #pu
        # pk      =  Sk_ph.real
        # phi_k   =  cmath.polar(Sk_ph)[1]
        # Psi_dk =  abs( (Sk_ph/Ykk_ph) - np.power(V_reg,2) ) 
        # Psi_pk =   pk  / Ykk * np.cos(phi_k + theta_kk)
        from copy import deepcopy
        Network_dummy = deepcopy(self)



        Psi_dk_dmm = abs(Network_dummy.Psi_d[i])
        Psi_pk_dmm = abs(Network_dummy.Psi_p[i])
        # Psi_qk_dmm = Network_dummy.Psi_q[i]

        Vmax = ((Psi_dk_dmm)**2 *0.5) + Psi_pk_dmm + np.sqrt((Psi_dk_dmm)**2*((Psi_dk_dmm)**2 +4*Psi_pk_dmm)   /2) 

        term_pos = np.power(V_reg,2)*(np.power(Psi_dk_dmm ,2) + 2*Psi_pk_dmm)
        term_neg = -np.power(V_reg,4) -  np.power(Psi_pk_dmm ,2) 



        # print('V_max',Vmax)
        # print('Pos',term_pos)
        # print('Neg',term_neg)
        q_re = -np.sqrt(abs(term_pos+term_neg ) )
        # print(pk,q_re)

        return q_re

        def ResetVars(self):
            Vk = Symbol('V_k')
            del_k = Symbol('\delta_{k}')
            Pk = Symbol('P_k')
            Qk = Symbol('Q_k')
            Sk = Symbol('S_k')
            Ykk = Symbol('Y_kk')
            Psi_pk = Symbol('\psi_{P.k}')
            Psi_qk = Symbol('\psi_{Q.k}')
            Psi_k = Symbol('\psi_{k}')
            Phi_k = Symbol('\phi_{k}')
            Theta_kk = Symbol('\Theta_{kk}')  
            ak = Symbol('alpha_k')
            bk = Symbol('beta_k')
            Dk = Symbol('D_k')
            Rc = Symbol('\gamma_k')
            delta_k = Symbol('\delta_k')
            phi_Dk  = Symbol('\phi_{Dk}')

            return Vk, del_k, Pk, Qk,Sk, Ykk, Psi_pk,Psi_qk, Psi_k, Phi_k, Theta_kk , ak, bk, Dk, Rc , delta_k , phi_Dk

    def truncate(self, n, decimals=0):
        multiplier = 10 ** decimals
        return int(n * multiplier) / multiplier

    def QSVS_system_profile(self, V_max, V_min, factor_load, off_load, show_data ):

        self.network.loads.p_set.iloc[:] = self.network.loads.p_set.iloc[:] * factor_load 
        self.network.loads.q_set.iloc[:] = self.network.loads.q_set.iloc[:] * factor_load 

        if off_load :
            for k in off_load:   
                if int(k)<=self.nBus:       
                    # print("Off-load at Bus :", k, 'Index:',int(k-2))
                    self.network.loads.p_set.iloc[int(k-2)] = 0
                    self.network.loads.q_set.iloc[int(k-2)] = 0 
            
        else :     
            pass  

        self.LoadFlow()

        QSVS = np.zeros(self.nBus)
        QC_keep = np.zeros(self.nBus)
        QC_snb_keep = np.zeros(self.nBus)
        VQ = np.zeros(self.nBus)
        VQ_Vmin= np.zeros(self.nBus)
        VQ_Vmax = np.zeros(self.nBus)
        Vmin = np.min( np.abs(self.V) )
        Psi_d_min_dmm = np.min( np.abs( self.Psi_d ))

        for i in range(self.nBus):
            # Filtering Parameter
            Psi_p_dmm = self.Psi_p[i]
            Psi_q_dmm = self.Psi_q[i]
            if Psi_p_dmm !=0 and Psi_q_dmm !=0:
                BusID = i+1

                #Revised Feb 4, 2022
                Vsnb = self.Vsnb(BusID)
                # print(np.angle(systemObj.Ybus[i,i]))

                QC_snb_dmm  = float(self.Qsnb( BusID ))
                
                QC_dmm      = float(self.Q_support( BusID))
                                
                VQ[i]=  sqrt( (Vsnb-abs(self.V[i]))**2  + ( float(QC_snb_dmm - QC_dmm ))**2 )
                
                # Update for minimum power loss
                QSVS[i] = VQ[i]
                # QSVS[i] = self.truncate(QSVS[i]  , 5)
            else :  
                VQ[i] = np.nan

                QSVS[i] = np.nan

                # print("Bus:",i+1,", VQ:",VQ[i])
                if show_data :
                    Vk = abs(self.V[i])

                    Qk = self.Bus.Q_pu[i]
                    
                    Ykk = self.Ybus[i,i]

            print("======= The "+str(BusID)+" bus==============")
            print("V : ",Vk )
            print("V-SNB : ",Vsnb )
            print("Qk (Current) : ",Qk )
            print("Qc (total-support)", QC_keep [i] )
            print("Qg (Reactive Generator Require at The maximum load ) : ", self.truncate( Vk**2 / np.imag(Ykk ),5))
            print("QSVS :",QSVS[i])

        return QSVS, VQ

    def Plot_VQ_curve(self, BusID, N_samples):

        # from copy import deepcopy
        # systemObj = deepcopy(self)

        Qsnb =  Symbol('Q^{SNB}_k')
        Vk, del_k, Pk, Qk,Sk, Ykk, Psi_pk,Psi_qk, Psi_dk, Phi_k, Theta_kk , ak, Dk, Rc , delta_k , phi_Dk = self.ResetVars()

        #Function VQ
        # Eqn9 = Vk**4 -(2*(Psi_qk/ak)+ (Psi_dk)**2)*(Vk**2) + (Psi_qk/ak)**2 + Psi_qk**2
        # sol = solveset(Eqn9 ,Psi_qk)
        # # pprint(sol)
        # # f_qk_pos = sol.args[0]
        # f_qk_pos = sol.args[1]

        #revised Feb 4, 2022
        # Psi_Qk_func= (Vk**2 * ak**2  -  sqrt(-Vk**2 * ak**2 + Psi_dk**2 * ak**2 + Psi_dk**2) )/(ak**2 + 1)
        Psi_Qk_func = sqrt(Vk * Psi_k**2  - (Vk**2 - Psi_pk)**2 ) 
        Qk_dmm_func = - Psi_Qk_func * Ykk * sin(Phi_k ) / sin(Phi_k+ Theta_kk)

        Qk_dmm_func = Qk_dmm_func.subs(Psi_k , abs(self.Psi_d[BusID-1]))
        Qk_dmm_func = Qk_dmm_func.subs( Psi_pk , self.Psi_p[BusID-1])
        Qk_dmm_func = Qk_dmm_func.subs(Ykk, abs(self.Ybus[BusID-1,BusID-1]) )
        Qk_dmm_func = Qk_dmm_func.subs(Phi_k, self.Phi[BusID-1])
        Qk_dmm_func = Qk_dmm_func.subs(Theta_kk, self.Theta[BusID-1])
        Qk_dmm_func = Qk_dmm_func.subs(ak, self.alpha[BusID-1])
        # Qk_dmm_func = Qk_dmm_func - self.Bus.Q_pu[BusID-1] 
        #Voltage at SNB
        # Vsnb = sqrt( (Psi_dk**2 )/ 8  + sqrt( (Psi_dk**4 /8) - (2*Psi_qk**2)  ) )
        # Vsnb = abs( N(Vsnb.subs(Psi_dk , abs(systemObj.Psi_d[BusID-1])).subs(Psi_qk , abs(systemObj.Psi_q[BusID-1]))))

        #Update Feb4,2022
        Vsnb = float( self.Vsnb(BusID) )
        # Qsnb = - Pk*Vk * Ykk * sqrt( -(2*Vk - Psi_dk)*( 2*Vk + Psi_dk) ) /Sk/ cos(Phi_k + Theta_kk) # pprint(Qsnb)


        #Display

        Vk_var = linspace(0,2,N_samples)

        func_QCk = lambdify(Vk, Qk_dmm_func, modules=['numpy'])
        QCsnb =  func_QCk(Vsnb)   

        fig, ax = plt.subplots( figsize=(12, 8))


        ax.plot(Vk_var , func_QCk(Vk_var) );
        ax.plot(Vsnb ,QCsnb  ,'o',label=': $Q^{SNB}_{'+str(BusID)+'}$');
        ax.xlabel = '$V$ (pu)'
        ax.ylabel = '$Q$ (pu)'
        ax.set_title('$Bus '+ str(BusID) +'$ on IEEE '+str(self.nBus)+'-bus test distribution system')
        ax.legend()

    def Optimal_size_RDGs( self,  busID, N_samples, Control, Q_P_Ratio, power_limit ):

        # Limitation : 0 - 100% of maximum demand P or/and Q     
        #Input: 
        # Bus ID
        # dimension

        #Output: 
            # CASE I: P-DG kW
            # CASE II: P-DG kW and Q-DG kVAr
            # CASE III: Q-DG kVAr
        # Control: P,Q,V control strategy for PF, must be “PQ”, “PV” or “Slack”.

            # https://pypsa.readthedocs.io/en/latest/components.html#generator


        # from copy import deepcopy
        org_system = deepcopy(self)   
        
        const = np.arange(0,N_samples+1,1) / N_samples

        XP = power_limit * const
        XQ = XP * Q_P_Ratio

        ind = np.arange(1,N_samples+2,1)
        pLoss = np.zeros(len(ind ))
        qLoss = np.zeros(len(ind ))
        # L =np.zeros(len(ind ))
        # QiVS = np.zeros(len(ind ))
        Gen_name = 'RDG'

        SizeRDGs_plot = XP     

        chk = 10000000000000000
        
        for i in tqdm( range(len(XP))  ,desc='Finding the optimal sizing by minimizing total loss' ): 
            
            systemObj = deepcopy(org_system) #Reset grid 
            systemObj.addGen(busID, XP[i], -XQ[i], Control, Gen_name )        
            systemObj.LoadFlow()

            # tmp_loss = float("{:.4f}".format(systemObj.total_kW_loss.astype(type('float', (float,), {}))))
            
            # tmp_loss = float("{:.4f}".format(systemObj.total_kW_loss.astype(type('float', (float,), {}))))
            pLoss[i]=round(systemObj.total_kW_loss,4)

            loadFactor = 1

            # L_tmp = np.amax( systemObj.L_index(loadFactor) )

            V_max = 1.05 
            V_min = 0.95  
            factor_load = 1 
            off_load = []
            show_data = false
            # QiVS_tmp =   round(min(systemObj.QiVS( V_max, V_min, factor_load, off_load,show_data)),4)
        
        del systemObj

        if (chk > pLoss[i]):
            OptimalSizeRDGs = XP[i]
            pLoss_min = pLoss[i]
            qLoss = np.nan
            # L = L_tmp
            # QiVS = QiVS_tmp 
            chk = pLoss[i]

        pLoss_plot = pLoss
        
        return OptimalSizeRDGs, pLoss_min, SizeRDGs_plot, pLoss_plot

    def Optimal_size_2RDGs( self,  busID, N_samples, Control, Q_P_Ratio, power_limit, speed=False):

        # Limitation : 0 - 100% of maximum demand P or/and Q     
        #Input: 
        # Bus ID
        # dimension

        #Output: 
            # CASE I: P-DG kW
            # CASE II: P-DG kW and Q-DG kVAr
            # CASE III: Q-DG kVAr
        # Control: P,Q,V control strategy for PF, must be “PQ”, “PV” or “Slack”.

            # https://pypsa.readthedocs.io/en/latest/components.html#generator


        # from copy import deepcopy
        org_system = deepcopy(self)   
        
        const = np.arange(0,N_samples+1,1) / N_samples

        XP1 = power_limit * const
        XQ1 = XP1 * Q_P_Ratio
        XP2 = power_limit * const
        XQ2 = XP2 * Q_P_Ratio

        ind = np.arange(1,N_samples+2,1)
        pLoss = np.zeros((len(ind ), len(ind )))
        qLoss = np.zeros((len(ind ), len(ind )))
        # L =np.zeros(len(ind ))
        # QiVS = np.zeros(len(ind ))
        Gen_name1 = 'RDG1'
        Gen_name2 = 'RDG2'
        # Control = 'PQ'
        SizeRDGs1_plot = XP1     
        SizeRDGs2_plot = XP2
        OptimalSizeRDG1 = []
        OptimalSizeRDG2 = []

        chk = 10000000000000000
        loss_org = org_system.total_kW_loss
        prv_loss = loss_org 
        if speed:
            print("You are using Speed mode!!")
        for i in tqdm( range(len(XP1))  ,desc='Finding the optimal sizing RDG 1 by minimizing total loss' ):
            for j in tqdm( range(len(XP2))  ,desc='Finding the optimal sizing RDG 2 by minimizing total loss' ): 
                
                if speed:
                    if prv_loss <= loss_org :
                        systemObj = deepcopy(org_system) #Reset grid 
                        systemObj.addGen(busID[0], XP1[i], XQ1[i], Control, Gen_name1 ) 
                        systemObj.addGen(busID[1], XP2[j], XQ2[j], Control, Gen_name2 )         
                        systemObj.LoadFlow()
                        pLoss[i,j]=systemObj.total_kW_loss
                        prv_loss = pLoss[i,j]

                    else : 
                        pLoss[i,j]=np.nan



                else :
                    systemObj = deepcopy(org_system) #Reset grid 
                    systemObj.addGen(busID[0], XP1[i], XQ1[i], Control, Gen_name1 ) 
                    systemObj.addGen(busID[1], XP2[j], XQ2[j], Control, Gen_name2 )         
                    systemObj.LoadFlow()
                    pLoss[i,j]=systemObj.total_kW_loss
                    # prv_loss = pLoss[i,j]

                    # tmp_loss = float("{:.4f}".format(systemObj.total_kW_loss.astype(type('float', (float,), {}))))
                    
                    # tmp_loss = float("{:.4f}".format(systemObj.total_kW_loss.astype(type('float', (float,), {}))))
                    
                
                
                    # del systemObj

                if (chk > pLoss[i,j]):
                    OptimalSizeRDG1 = XP1[i]
                    OptimalSizeRDG2 = XP2[j]
                    pLoss_min = pLoss[i,j]
                    # qLoss = np.nan
                    # L = L_tmp
                    # QiVS = QiVS_tmp 
                chk = pLoss[i,j]

            
        
        return OptimalSizeRDG1, OptimalSizeRDG2,pLoss_min, SizeRDGs1_plot, SizeRDGs2_plot,  pLoss

    def Vsnb(self, BusID):
        # Voltage at SNB of Bus no. i : V_SNB(i)
        # Revision date: APR 13, 2022
        systemObj = deepcopy(self)
        Psi_k_dmm =np.abs(systemObj.Psi_d[BusID -1])
        V_k_dmm = abs(systemObj.V[BusID -1])
        Psi_qk_dmm = systemObj.Psi_q[BusID -1]
        Xi_dmm = np.angle(systemObj.Psi_d[BusID -1])
        delta_dmm = np.angle(systemObj.V[BusID -1])
        Delta_k_snb = np.arccos(-Psi_k_dmm/ V_k_dmm /2)

        # Psi_qk_snb = np.cos(delta_dmm)*2*V_k_dmm**2*sqrt(1-np.cos(delta_dmm)**2 )
        Delta_k_temp =  Xi_dmm-delta_dmm 

        dmm =abs( 4*V_k_dmm**2*np.cos(Delta_k_temp)**2 - Psi_k_dmm**2)
        Vsnb = sqrt( V_k_dmm**2 - sqrt( V_k_dmm**2 *(dmm)  * sin(Delta_k_temp )**2 ))    
        return Vsnb

    def Qc_snb(self, BusID):
        # Reactive support at SNB of Bus no. i : Qc_SNB(i)
        # Revision date: APR 13, 2022

        Vk_dmm    = np.abs(self.V[ BusID-1])
        Phi_dmm   = np.angle(self.S[ BusID-1] )
        Xi_dmm    = np.abs(self.Psi_q[BusID -1])
        delta_dmm = np.angle(self.V[ BusID-1])
        Delta_dmm = Xi_dmm - delta_dmm 
        Vsnb_dmm =  self.Vsnb( BusID)
        Ykk       = np.abs(self.Ybus[ BusID-1, BusID-1 ])
        Theta_kk  = np.angle(self.Ybus[ BusID-1, BusID-1 ])
        Psi_p_dmm = np.abs( self.Psi_p[BusID -1] )

        Psi_k_dmm =np.abs(self.Psi_d[BusID -1])
        Psi_q_SNB_dmm = np.sign( sin(Delta_dmm))* sqrt(abs(np.power(Vsnb_dmm,2)*np.power(Psi_k_dmm ,2)   - ((Vsnb_dmm**2 - Psi_p_dmm  )**2)))
        # print("Psi_Q snb : ",Psi_q_SNB_dmm)
        Qsnb = - Ykk * sin(Phi_dmm  )  * Psi_q_SNB_dmm / sin(Phi_dmm + Theta_kk)

        return Qsnb

    def Qc(self, BusID):
        # Reactive support at Bus no. i : Qc(i)
        # Revision date: APR 13, 2022

        Phi_dmm   = np.angle(self.S[ BusID-1] )
        Ykk       = np.abs(self.Ybus[ BusID-1, BusID-1 ])
        Theta_kk  = np.angle(self.Ybus[ BusID-1, BusID-1 ])
        Psi_q_dmm = np.abs( self.Psi_q[BusID -1] )
        Psi_k_dmm =np.abs(self.Psi_d[BusID -1])

        # Psi_q_dmm = np.sign( sin(Delta_dmm))* sqrt( np.power(Vk_dmm ,2)*np.power(Psi_k_dmm ,2)   - ((Vk_dmm **2 - Psi_p_dmm  )**2))
        Qc = - Ykk * np.sin(Phi_dmm) * Psi_q_dmm / np.sin(Phi_dmm + Theta_kk)
        return Qc

    def VQ_profile(self, factor_load, off_load, show_data ):

        # Revision date: APR 13, 2022

        systemObj = deepcopy(self)
        systemObj.network.loads.p_set.iloc[:] = systemObj.network.loads.p_set.iloc[:] * factor_load 
        systemObj.network.loads.q_set.iloc[:] = systemObj.network.loads.q_set.iloc[:] * factor_load 
        VQ_max = 0
        if len(off_load) :
            for k in list( off_load ):    
            
                systemObj.network.loads.p_set[systemObj.network.loads.bus == "Bus No "+str(k)] = 0
                systemObj.network.loads.q_set[systemObj.network.loads.bus == "Bus No "+str(k)] = 0
            
        else :     
            pass  

        systemObj.LoadFlow()

        QC_keep = np.zeros(systemObj.nBus)
        QC_snb_keep = np.zeros(systemObj.nBus)
        VQ = np.zeros(systemObj.nBus)
        # VQ_Vmin= np.zeros(systemObj.nBus)
        # VQ_Vmax = np.zeros(systemObj.nBus)

        for i in range(systemObj.nBus):
            # Filtering Parameter
            BusID = i+1
            Psi_p_dmm = abs(systemObj.Psi_d[i])
            Psi_q_dmm = abs(systemObj.Psi_q[i])
            Psi_k_dmm = abs(systemObj.Psi_d[i])
            Vsnb_dmm = systemObj.Vsnb( BusID )
            Vk_dmm = abs(systemObj.V[i])
            
            QC_snb_dmm  = systemObj.Qc_snb(BusID)
            
            QC_dmm      = systemObj.Qc(BusID)

            VQ_dmm =  sqrt( (Vsnb_dmm-Vk_dmm)**2  + ( QC_snb_dmm - QC_dmm )**2 )

            if QC_dmm!=0 :

            

            #Revised Feb 4, 2022
            # Vsnb_dmm = systemObj.Vsnb( BusID )
            # Vk_dmm = abs(systemObj.V[i])
            
            # QC_snb_dmm  = systemObj.Qc_snb(BusID)
            
            # QC_dmm      = systemObj.Qc(BusID)

                try:
                    VQ[i]= sqrt( (Vsnb_dmm-Vk_dmm)**2  + ( QC_snb_dmm - QC_dmm )**2 )
                

                except :
                    print("VQ at bus :",i+1," is infeasible to solve with off load(s) : ",off_load)
                    print("Vsnb :",Vsnb_dmm)
                    print("V :",Vk_dmm)
                    print("QC_snb :",QC_snb_dmm)
                    print("QC :",QC_dmm)
                    
                    # VQ[i] = np.nan

                    
                
                    # QSVS[i] =  systemObj.truncate(QSVS[i]  , 5)
            else :  
                VQ[i] = np.nan

            

            # print("Bus:",i+1,", VQ:",VQ[i])
            if show_data :
                Vk = abs(systemObj.V[i])

                Qk = systemObj.Bus.Q_pu[i]
                
                Ykk = systemObj.Ybus[i,i]

                print("======= The "+str(BusID)+" bus==============")
                print("dist(VQ) : ",VQ_dmm  )
                print("Vk: ",Vk_dmm )
                print("V-SNB : ",Vsnb_dmm )
                print("QC (Current) : ",QC_dmm)
                print("QC_snb :",QC_snb_dmm)

                del Psi_p_dmm, Psi_q_dmm, Psi_k_dmm, Vsnb_dmm, Vk_dmm, QC_snb_dmm ,QC_dmm,VQ_dmm 

            # print("Qc (total-support)", QC_dmm )
            # print("Qg (Reactive Generator Require at The maximum load ) : ", systemObj.truncate( Vk**2 / np.imag(Ykk ),5))
            # print("QSVS :",QSVS[i])
            del systemObj  
        return VQ

    def V_func(self, SystemObj_input, P_var, RCR , BusID ):

        tolerance = 10000
        systemObj = deepcopy(self)

        k = BusID
        if int( k )<= systemObj.nBus:       
            # print("Off-load at Bus :", k, 'Index:',int(k-2))
            systemObj.network.loads.p_set.iloc[int(k-2)] = P_var
            systemObj.network.loads.q_set.iloc[int(k-2)] = P_var * RCR 
            systemObj.LoadFlow()
            
            #Getting parameters
            Psi_pk = abs( systemObj.Psi_p[BusID-1] )
            Psi_dk = abs(  systemObj.Psi_d[BusID-1] )
            Psi_qk = abs(  systemObj.Psi_q[BusID-1] )
            alpha_k = abs(  systemObj.alpha[BusID-1] ) 
            Phi_k =  systemObj.Phi[BusID-1] 
            Theta_kk = np.angle( systemObj.Ybus[BusID-1,BusID-1] ) 
            Y_kk = abs( systemObj.Ybus[BusID-1,BusID-1] ) 
            try :
                Vsnb = sqrt( (Psi_dk**2 )/ 8  + sqrt( (Psi_dk**4 /8) - (2*Psi_qk**2)  ) )

                V = sqrt( Psi_dk**4 + ( 4 * (Psi_dk**2 )*Psi_pk) - (4 * Psi_pk**2 * alpha_k**2 ) )
                V = sqrt( (Psi_dk**2 / 2 )+ Psi_pk + (V/2) )

                if Vsnb > tolerance :
                    print("Voltage Collapse at P_load = ",P_var)
                    print("Voltage Collapse at Q_load = ",P_var * RCR)

            
            except :
                # print("Voltage Collapse at P_load = ",P_var)
                # print("Voltage Collapse at Q_load = ",P_var * RCR)
                Vsnb = tolerance*10
                V = tolerance*10

        return V, Vsnb

    def Loading_margin(self, RCR_input , BusID_input , P_var_limit=10,sensitivity = 0.01):

        systemObj_dmm = deepcopy(self)
        p_var =  np.arange(0.00001, P_var_limit, sensitivity)
        V_var =  np.zeros( len(p_var) ) 
        Vsnb = 0
        i = 0
        tolerance = 10000
        while  Vsnb < tolerance  and i < len(p_var):
            
            V_temp = self.V_func( systemObj_dmm, p_var[i], RCR_input , BusID_input )
            V_var[i] = V_temp [0]
            Vsnb = V_temp [1]
            # print("P_var[i] :", p_var[i])
            # print("V_var[i] :", V_temp [0])
            # print("Vsnb: ",V_temp [1])

            i = i+1

        print("--------- Voltage collapse ---------")
        print("At Bus : ", BusID_input)
        print("V snb : ", Vsnb )
        print("V (before)  : ",V_var[i-1] )
        print("Loading margin : ",p_var[i-1] )

        return p_var[i-1]

    def voltage_product_profile(self, factor_load = [1], off_load = [] ):

        # Revision date: APR 13, 2022
        systemObj = deepcopy(self)

        systemObj.network.loads.p_set.iloc[:] = systemObj.network.loads.p_set.iloc[:] * factor_load 
        systemObj.network.loads.q_set.iloc[:] = systemObj.network.loads.q_set.iloc[:] * factor_load 

        if off_load :
            for k in off_load:   
                if int(k)<=systemObj.nBus:       
                    # print("Off-load at Bus :", k, 'Index:',int(k-2))
                    systemObj.network.loads.p_set[systemObj.network.loads.bus== "Bus No "+str(k)] = 0
                    systemObj.network.loads.q_set[systemObj.network.loads.bus== "Bus No "+str(k)] = 0
                    # systemObj.network.loads.p_set.iloc[int(k-2)] = 0
                    # systemObj.network.loads.q_set.iloc[int(k-2)] = 0     
                else :     
                    pass 

        systemObj.LoadFlow()

        return abs(systemObj.Psi_d[:]) 

    def voltage_profile(self, factor_load = [1], off_load = [] ):
        
        # Revision date: APR 13, 2022
        systemObj = deepcopy(self)
        
        systemObj.network.loads.p_set.iloc[:] = systemObj.network.loads.p_set.iloc[:] * factor_load 
        systemObj.network.loads.q_set.iloc[:] = systemObj.network.loads.q_set.iloc[:] * factor_load 

        if off_load :
            for k in off_load:   
                if int(k)<=systemObj.nBus:       
                    # print("Off-load at Bus :", k, 'Index:',int(k-2))
                    systemObj.network.loads.p_set[systemObj.network.loads.bus== "Bus No "+str(k)] = 0
                    systemObj.network.loads.q_set[systemObj.network.loads.bus== "Bus No "+str(k)] = 0
                    # systemObj.network.loads.p_set.iloc[int(k-2)] = 0
                    # systemObj.network.loads.q_set.iloc[int(k-2)] = 0     
        else :     
            pass 

        systemObj.LoadFlow()

        return abs(systemObj.V[:])

if __name__ == "__main__" :

#Query from CSV
    # MVA_base = 100.0 # Defining the Base-MVA
    # KV_base = 100 #KV 
    # nBus = 5
    # SLACK_POS = 1
    # NETWORK_ = ''
    # IEEE5_csv = Grid( nBus, NETWORK_, 'faifah/5BusTest-line.csv', 'faifah/5BusTest-load.csv', MVA_base ,  KV_base, SLACK_POS)
    # # ( self, nbus, NETWORK_ = '', LINE_DATA_CSV='', BUS_DATA_CSV='', MVA_base = np.nan , KV_base = np.nan, SLACK_POS = np.nan ):


    # #Adding gen for IEEE 5-Bus test system
    # BusID = 2
    # P_KW = 40
    # Q_KVAr = 30
    # Control='PQ'
    # Gen_name = "G2"
    # IEEE5.addGen(BusID, P_KW, Q_KVAr, Control,Gen_name)
    # IEEE5.LoadFlow()
    # IEEE5_csv.Report()

#Query from package
    NETWORK_ = 'IEEE5'
    MVA_base = 100.0 # Defining the Base-MVA
    KV_base = 100 #KV 
    nBus = 5
    SLACK_POS = 1

    IEEE5 = Grid( nBus, NETWORK_, '', '', MVA_base ,  KV_base, SLACK_POS )
    IEEE5.Report()

    # print( IEEE5.line_data )
    # print( IEEE5.load_data )



    



