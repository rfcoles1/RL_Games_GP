import numpy as np
import GPy
import matplotlib.pyplot as plt

LatentHeat_Fusion = 334 #kJ/kg
LatentHeat_Vapor = 2264.705 #kJ/kg
MeltingPoint = 0 #degC
BoilingPoint = 100.0 #degC
HeatCap_Ice = 2.108 #kJ/kg/C
HeatCap_Water = 4.148 #kJ/kg/C
HeatCap_Steam = 1.996 #kJ/kg/C

class Engine(object):
    def __init__(self, T = 21, M = 1): #must initalise at a temperature, not a mass fraction
        self.Ti = T 
        self.RoomTemp = 21 
        #if starting not at room temp, initial value for Energy in will be the energy taken to get there 
        self.Mass = M #mass is not implemented yet 
        
        self.gen_critPoints() #calculates energy at which transitions begin/end
        self.reset()

    def reset(self):
        self.T = self.Ti
        self.Terr = 0
        
        self.MassFractions = np.array([0,1,0]) #assumes initial is all water, change as appropriate
        self.MFrac =  self.encodeMass(self.MassFractions)  
        self.Merr = 0
        
        self.EnergyIn = self.get_Energy_From_Temp(self.T)
        
        self.input_memory = []
        self.output_memory_T = []
        self.output_memory_M = []   
        
        self.add_data()

    #generates the models used for Temperature and Mass Fraction prediction
    def gen_model(self):
        kern = GPy.kern.RBF(input_dim = 1, variance = .1, lengthscale=200.0) 
        self.TempModel = GPy.models.GPRegression(np.array(self.input_memory).reshape(-1,1),\
            np.array(self.output_memory_T).reshape(-1,1), kern)
        self.TempModel.Gaussian_noise.variance = 0 #data is true, no noise 
        self.TempModel.Gaussian_noise.variance.fix()
        kern.lengthscale.fix()
        self.TempModel.optimize()
        
        kern = GPy.kern.RBF(input_dim = 1, variance = .1, lengthscale=200.0)
        self.MassModel = GPy.models.GPRegression(np.array(self.input_memory).reshape(-1,1),\
            np.array(self.output_memory_M).reshape(-1,1), kern)
        self.MassModel.Gaussian_noise.variance = 0
        self.MassModel.Gaussian_noise.variance.fix()
        kern.lengthscale.fix()
        self.MassModel.optimize()
        
    def add_data(self):
        self.input_memory = np.concatenate([self.input_memory, [self.EnergyIn]])
        output = self.get_true_value(self.EnergyIn)
        self.output_memory_T = np.concatenate([self.output_memory_T, [output[0]]])
        self.output_memory_M = np.concatenate([self.output_memory_M, [self.encodeMass(output[1])]])
        self.gen_model() 
        
    def get_state(self):
        tempM = np.vstack([self.MFrac, self.Merr])
        tempT = np.vstack([self.T, self.Terr])
        return np.hstack([tempT, tempM])

    def get_pred(self, inp):
        return self.TempModel._raw_predict(inp), self.MassModel._raw_predict(inp)
        
        
    #encode the mass fraction as a single number
    #[1,0,0] -> 0 all ice
    #[0,1,0] -> 1 all water
    #[0,0,1] -> 2 all steam
    
    #[0.5,0.5,0] -> 0.5 half way between ice and water
    #[0,0.3,0.7] -> 1.3 30% between water and steam 
    #and so on
    def encodeMass(self, m):
        m = np.array(m)
        tmp = np.where(m>0)[0]
        maxval = np.max(tmp)
        return(maxval - 1 + m[maxval])
    
    def decodeMass(self, M):
        massfrac = np.zeros(3)
        mInd = int(np.ceil(M))
        if M%1 != 0:
            massfrac[mInd] = M%1
            massfrac[mInd - 1] = 1-M%1
        else:
            massfrac[mInd] = 1
        return massfrac
     
     
    def get_true_value(self, E): #gives true values for temperature and mass fraction, given an energy value
        if E > self.Upper_Boiling_Energy:
            T = BoilingPoint + (1./HeatCap_Steam)*(E-self.Upper_Boiling_Energy)
            MassFractions = [0,0,1]
            return T, MassFractions
        elif E > self.Lower_Boiling_Energy:
            T = BoilingPoint
            Ediff = E - self.Lower_Boiling_Energy
            MassFractions = [0, 1. - Ediff/LatentHeat_Vapor, Ediff/LatentHeat_Vapor]
            return T, MassFractions
        elif E > self.Upper_Melting_Energy:
            T = MeltingPoint + (1./HeatCap_Water)*(E-self.Upper_Melting_Energy)
            MassFractions = [0,1,0]
            return T, MassFractions
        elif E > self.Lower_Melting_Energy:
            T = MeltingPoint
            Ediff = E - self.Lower_Melting_Energy
            MassFractions = [1. - Ediff/LatentHeat_Fusion, Ediff/LatentHeat_Fusion, 0]
            return T, MassFractions
        else:
            T = MeltingPoint + (1./HeatCap_Ice)*(E - self.Lower_Melting_Energy)
            MassFractions = [1,0,0]
            return T, MassFractions

    def get_Energy_From_Temp(self, T): #wont be necessarily correct for Boiling/Melting temperature 
        if T < MeltingPoint:
            thisE = self.Lower_Melting_Energy + -(abs(T-MeltingPoint)*HeatCap_Ice)
        elif T < BoilingPoint:
            thisE = self.Upper_Melting_Energy + ((T - MeltingPoint)*HeatCap_Water)
        else:   
            thisE = self.Upper_Boiling_Energy + (abs(T-BoilingPoint)*HeatCap_Steam)
        return thisE
            
    def gen_critPoints(self): #calculates energy at which transitions begin/end
        self.Upper_Melting_Energy = (MeltingPoint - self.RoomTemp) * HeatCap_Water
        self.Lower_Melting_Energy = self.Upper_Melting_Energy - LatentHeat_Fusion

        self.Lower_Boiling_Energy = (BoilingPoint - self.RoomTemp) * HeatCap_Water
        self.Upper_Boiling_Energy = self.Lower_Boiling_Energy + LatentHeat_Vapor
        
           
    def plt_model(self): #plots the current model around collected data points 
        Energy = np.linspace(min(self.input_memory) -100, max(self.input_memory) + 100, 100).reshape(-1,1)
        TempMeans = np.zeros(len(Energy))
        TempSdvs = np.zeros(len(Energy))
        Temp = np.zeros(len(Energy))
        MassMeans = np.zeros(len(Energy))
        MassSdvs = np.zeros(len(Energy))
        Mass = np.zeros(len(Energy))
        for i in range(len(Energy)):
            out = self.TempModel._raw_predict(np.array([Energy[i]]).reshape(-1,1))
            TempMeans[i] = out[0].flatten()[0]
            TempSdvs[i] = out[1].flatten()[0]
            Temp[i] = self.get_true_value(Energy[i])[0]
            
            out = self.MassModel._raw_predict(np.array([Energy[i]]).reshape(-1,1))
            MassMeans[i] = out[0].flatten()[0]
            MassSdvs[i] = out[1].flatten()[0]
            Mass[i] = self.encodeMass(self.get_true_value(Energy[i])[1])
            
        plt.figure()
        plt.subplot(121)
        plt.plot(Energy, Temp, 'k', label = 'True')
        plt.plot(Energy, TempMeans, label = 'Mean')
        plt.fill_between(Energy.flatten(), TempMeans - TempSdvs, TempMeans + TempSdvs, facecolor = 'r', alpha = 0.5, label = '1 sigma range')
        plt.ylabel('Temperature ($^\circ$C)')
        plt.xlabel('Energy Added (kJ)')
        plt.ylim(top = max(Temp) +10, bottom = min(Temp) - 10)
        plt.legend()
        
        plt.subplot(122)
        plt.plot(Energy, Mass, 'k', label = 'True')
        plt.plot(Energy, MassMeans, label = 'Mean')
        plt.fill_between(Energy.flatten(), MassMeans - MassSdvs, MassMeans + MassSdvs, facecolor = 'r', alpha = 0.5, label = '1 sigma range')
        plt.ylabel('MassFractions')
        plt.xlabel('Energy Added (kJ)')
        plt.ylim(top = max(Mass) + 0.1, bottom = min(Mass) - 0.1)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        
    def plt_AllTemp(self, minT = -50, maxT = 150): #plots the temperature changes against energy in 
        if minT >= maxT:
            print("minT > maxT")
            return

        minE = self.get_Energy_From_Temp(minT)
        maxE = self.get_Energy_From_Temp(maxT)
            
        Energy = np.linspace(minE, maxE, 100)
        Temp = np.zeros(len(Energy))
        for i in range(len(Energy)):
            Temp[i] = self.get_true_value(Energy[i])[0]
            
        plt.plot(Energy, Temp, 'k', label = 'True')
        plt.xlim(left = minE, right = maxE)
        plt.ylim(bottom = minT, top = maxT)
        plt.ylabel('Temperature ($^\circ$C)')
        plt.xlabel('Energy Added (kJ)')
        plt.tight_layout()
        plt.show()

    def plt_AllMass(self, minT = -50, maxT = 150): #plots the massfraction changes against energy in 
        if minT >= maxT:
            print("minT > maxT")
            return

        minE = self.get_Energy_From_Temp(minT)
        maxE = self.get_Energy_From_Temp(maxT)
            
        Energy = np.linspace(minE, maxE, 100)
        Mass = np.zeros(len(Energy)) 
        for i in range(len(Energy)):
            Massfrac = self.get_true_value(Energy[i])[1]
            Mass[i] = self.encodeMass(Massfrac)
        plt.plot(Energy, Mass, 'k', label = 'True')
        plt.xlim(left = minE, right = maxE)
        plt.ylim(bottom = -0.1, top = 2.1)
        plt.yticks([0,0.5,1,1.5,2],['All Ice', '50% Ice/Water', 'All Water', '50% Water/Steam','All Steam'])
        plt.xlabel('Energy Added (kJ)')
        plt.ylabel('Mass Fractions')
        plt.tight_layout()
        plt.show()

        