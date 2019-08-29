import numpy as np
import GPy

import matplotlib.pyplot as plt

height_scale = 10
delta = 0.025

def Gibbs(temperature, pressure):
    Z1 = np.exp(-temperature**2 - pressure**2)
    Z2 = np.exp(-(temperature - 1)**2 - (pressure - 1)**2)
    return height_scale*(Z1 - Z2) * 2


def phase_check(gibbs):
    if gibbs   < -15:
        phase = 'i'
    elif gibbs < -10:
        phase = 'ii'
    elif gibbs <  -5:
        phase = 'iii'
    elif gibbs <   0:
        phase = 'iv'
    elif gibbs <   5:
        phase = 'v'
    elif gibbs <  10:
        phase = 'vi'
    elif gibbs <  15:
        phase = 'vii'
    else:
        phase = 'viii'
    return phase


dGdT = { 'i'     : .1,
         'ii'    : .2,
         'iii'   : .3,
          'iv'   : .4,
          'v'    : .5,
          'vi'   : .6,
          'vii'  : .7,
          'viii' : .8
       }

dGdP = { 'i'     : 1.,
         'ii'    : 2.,
         'iii'   : 3.,
          'iv'   : 4.,
          'v'    : 5.,
          'vi'   : 6.,
          'vii'  : 7.,
          'viii' : 8.
       }

class Engine(object):
    def __init__(self, Ti = 0, Pi = 0):
        self.T = Ti
        self.P = Pi
        self.G = Gibbs(Ti, Pi)
        self.G_err = 0
        self.phase = phase_check(self.G)
        self.reset()

    def reset(self, T = 0, P = 0):
        
        self.reset_history()

        self.T = T
        self.P = P

        self.input_memory = []
        self.output_memory = []
        self.add_data()
       
            
    def reset_history(self):
        self.T_history = []
        self.P_history = []
        self.process_history = [] # tracks dQ and dW into and out of the sample
        self.process_history_letters = ''
        self.Q_history = [] # heat which has been applied
        self.W_history = [] # work which has been applied   

    def get_state(self):
        return np.array([self.G, self.G_err])
   
    def get_true(self, T, P):
        G = Gibbs(T,P)
        ph = phase_check(G)
        return G, ph

    def set_temperature(self, T):
        self.T = T
        self.update_state()

    def set_pressure(self, P):
        self.P = P
        self.update_state()

    def update_state(self):
        tmp = self.get_pred([self.T, self.P])
        self.G = tmp[0].flatten()[0]
        self.G_err = tmp[1].flatten()[0]

        self.phase = phase_check(self.G)

        self.T_history.append(self.T)
        self.P_history.append(self.P)
 
    def gen_model(self):
        kern = GPy.kern.RBF(input_dim = 2, variance = .1, lengthscale=1.0)
        self.model = GPy.models.GPRegression(np.array(self.input_memory).reshape(len(self.input_memory)/2,2),\
            np.array(self.output_memory).reshape(-1,1), kern)

        self.model.Gaussian_noise.variance = 0
        self.model.Gaussian_noise.variance.fix()
        kern.lengthscale.fix()
        self.model.optimize()

    def add_data(self):
        self.input_memory = np.concatenate([self.input_memory, [self.T, self.P]])
        output = Gibbs(self.T, self.P)
        self.output_memory = np.concatenate([self.output_memory, [output]])
        self.gen_model()
        self.update_state()

    def get_pred(self, inp):
        inp = np.array(inp).reshape(len(inp)/2,2)
        return self.model._raw_predict(inp)
    
    def apply_heat(self, dQ):
        self.Q_history.append(dQ)
        self.process_history.append([dQ,0]) # add one-hot in the Q column
        if dQ > 0:
          # positive heat means it is heated
          self.process_history_letters += 'h'
        else:
          # it is cooled
          self.process_history_letters += 'c'
        dT = dQ/dGdT[self.phase]
        self.T += dT
        self.update_state()

    def apply_work(self, dW):
        self.W_history.append(dW)
        self.process_history.append([0,dW]) # add one-hot in the W column
        if dW > 0:
          # positive work squishes it
          self.process_history_letters += 's'
        else:
          # negative work means it expands
          self.process_history_letters += 'e'
        dP = dW/dGdP[self.phase]
        self.P += dP
        self.update_state()


    def plt_path(self):
        T, P = self.T_history, self.P_history

        fig, ax = plt.subplots()
        cmap = plt.get_cmap("plasma")
        ax.plot(T, P)

        x = np.arange(-3.0, 3.0, delta)
        y = np.arange(-3.0, 3.0, delta)
        X, Y = np.meshgrid(x,y)
        Z = Gibbs(X,Y)
        CS = ax.contour(X, Y, Z, cmap=cmap)

        ax.clabel(CS, inline=1, fontsize=10)
        ax.set_title('Phase diagram / Gibbs')

        ax.set_xlabel('Temperature')
        ax.set_ylabel('Pressure')

        plt.show()

    def plt_model(self):
        
        plt.figure()
        
        cmap = plt.get_cmap("plasma")
        
        x = np.linspace(-3.0, 3.0, 100)
        y = np.linspace(-3.0, 3.0, 100)
        X, Y = np.meshgrid(x,y)

        Z = np.zeros([len(x),len(y)])
        Z_Err = np.zeros([len(x),len(y)])
        TrueZ = np.zeros([len(x),len(y)])
        
        for i in range(len(x)):
            for j in range(len(y)):

                out = self.get_pred([x[i],y[j]])
                Z[i][j] = out[0].flatten()[0]
                Z_Err[i][j] = out[1].flatten()[0]
                TrueZ[i][j] = Gibbs(x[i],y[j])

        plt.subplot(221)
        plt.imshow(Z, origin = 'lower')   

        plt.subplot(222)
        T, P = self.T_history, self.P_history
        plt.plot(T, P)
        CS = plt.contour(X, Y, TrueZ, cmap=cmap)

        plt.subplot(223)
        plt.imshow(Z_Err, origin = 'lower')

        plt.subplot(224)
        plt.imshow(TrueZ, origin = 'lower') 

        plt.show()
