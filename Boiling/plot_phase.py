import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})


#Triple Point
T_tri = 273.16 #K
P_tri = 0.61173 #kPa

# P = 1 atm boiling point
T_boil = (100 + 273.15) #K
P_boil = 100.0 #kPa

#critical point
T_crit = 647.3 #K
P_crit = 22050.0 #MPa


#enthalpy, entropy, volume (molar)
H_melt = 5.98 #kJ/mol
S_melt = 0.022 #kJ/mol*K
V_melt = -1.634e-6 #m3/mol

H_vap = 44.9 #kJ/mol
S_vap = 0.165 #kJ/mol*K
V_vap = 0.022050 #m3/mol

H_sub = 50.9 #kJ/mol
S_sub = 0.168 #kJ/mol*K
V_sub = 0.022048 #m3/mol

R = 0.008314 #kJ/mol*K

T_melt = np.linspace(T_tri - 10.0, T_tri, 100)
P_melt = P_tri + H_melt/V_melt*np.log(T_melt/T_tri)

T_sub = np.linspace(T_tri - 150.0, T_tri, 100)
P_sub = P_tri * np.exp((H_sub/R)*((1/T_tri) - (1/T_sub)))

T_vap = np.linspace(T_tri, T_crit, 100)
P_vap = P_tri * np.exp((H_vap/R)*((1/T_tri) - (1/T_vap)))


area_alpha = 0.1
line_alpha = 0.7

fig = plt.figure()
plt.scatter([-50 + 273.15,150 +  273.15] ,[101,101], color = 'k', marker = 'D', s = 5)
plt.plot(T_sub, P_sub, color = 'lime', alpha = line_alpha, label = 'Sublimation')
plt.plot(T_melt,P_melt, color = 'r', alpha = line_alpha, label = 'Melting')
plt.plot(T_vap, P_vap, color = 'c', alpha = line_alpha, label = 'Evaporation')
plt.plot([-50 + 273.15,150 +  273.15] ,[101,101], color = 'k')
plt.scatter(T_tri, P_tri, color = 'k', zorder = 10)

plt.yscale('log')
plt.ylim(bottom = 1e-4, top = 1e5)
plt.xlim(left = 150, right = 625)
plt.ylabel('Pressure')
plt.yticks([1e-3,1,1e3], ['1 Pa', '1 kPa', '1 MPa'])
#plt.xlabel('Temperature (K)')
#plt.xticks(np.arange(200, 625, 100))
plt.xlabel('Temperature ($^\circ$C)')
plt.xticks(np.arange(173.15, 574, 100), np.arange(-100, 301, 100))

#solid
#plt.fill_between(T_melt, , P_melt, alpha = alpha, color = 'y')
plt.fill_between(T_sub, P_sub, 1e6, alpha = area_alpha, color = 'y')
plt.fill_between(T_melt, P_melt, 1e6, color = 'w')#remove color from area above Sublimation and Melting

plt.fill_between(np.concatenate([T_melt, T_vap]), np.concatenate([P_melt, P_vap]), 1e6,\
    alpha = area_alpha, color = 'm') #liquid
plt.fill_between(np.concatenate([T_sub, T_vap]), 0, np.concatenate([P_sub, P_vap]),\
    alpha = area_alpha, color = 'g') #gas

plt.annotate('Solid', xy=(170, 1.2), xycoords = 'data', fontsize = 'large', color = 'darkgoldenrod')
plt.annotate('Liquid', xy = (330, 3e3), xycoords = 'data', fontsize = 'large', color = 'purple')
plt.annotate('Gas', xy = (450, 5e-1), xycoords = 'data', fontsize = 'large', color = 'seagreen')

plt.legend(loc = 'lower right')
plt.tight_layout()
plt.show()
