import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import sys
sys.path.insert(0,'./V1')
from boil_engine import Engine

LatentHeat_Fusion = 334 #kJ/kg
LatentHeat_Vapor = 2264.705 #kJ/kg
MeltingPoint = 0 #degC
BoilingPoint = 100.0 #degC
HeatCap_Ice = 2.108 #kJ/kg/C
HeatCap_Water = 4.148 #kJ/kg/C
HeatCap_Steam = 1.996 #kJ/kg/C

Eng = Engine()

minT = -50
maxT = 150
minE = Eng.get_Energy_From_Temp(minT)
maxE = Eng.get_Energy_From_Temp(maxT)

Energy = np.linspace(minE, maxE, 100)
Temp = np.zeros(len(Energy))
Mass = np.zeros(len(Energy))
for i in range(len(Energy)):
    Temp[i] = Eng.get_true_value(Energy[i])[0]
    Mass[i] = Eng.encodeMass(Eng.get_true_value(Energy[i])[1])

transitions = [Eng.Lower_Melting_Energy, Eng.Upper_Melting_Energy, \
    Eng.Lower_Boiling_Energy, Eng.Upper_Boiling_Energy]

plt.figure()

line_al = 0.4
shade_al = 0.05

def section_plot(xlim, ylim):
    plt.plot([transitions[0], transitions[0]], ylim, alpha = line_al, linestyle = 'dashed', color = 'darkgoldenrod' )
    plt.plot([transitions[1], transitions[1]], ylim, alpha = line_al, linestyle = 'dashed', color = 'purple')
    plt.plot([transitions[2], transitions[2]], ylim, alpha = line_al, linestyle = 'dashed', color = 'blue')
    plt.plot([transitions[3], transitions[3]], ylim, alpha = line_al, linestyle = 'dashed', color = 'seagreen')
    
    plt.axvspan(xlim[0], transitions[0], alpha = shade_al, color = 'y')
    plt.axvspan(transitions[0], transitions[1], alpha = shade_al, color = 'r')
    plt.axvspan(transitions[1], transitions[2], alpha = shade_al, color = 'm')
    plt.axvspan(transitions[2], transitions[3], alpha = shade_al, color = 'c')
    plt.axvspan(transitions[3], xlim[1], alpha = shade_al, color = 'g')

gs = gridspec.GridSpec(2,1)
gs.update(hspace = 0)
xticks = np.arange(-500, 2501, 500)

ax1 = plt.subplot(gs[0])
plt.plot(Energy, Temp, 'k')
plt.ylabel('Temperature ($^\circ$C)', labelpad = 5)
plt.xlabel('Energy Added (kJ)')
plt.xlim(left = minE, right = maxE)
plt.ylim(bottom = minT-20, top = maxT+20)
plt.xticks(xticks, [])
plt.tick_params('x', direction = 'in')
ax1.get_yaxis().set_label_coords(-0.065,0.5)

ylim = ax1.get_ylim()
xlim = ax1.get_xlim()
section_plot(xlim, ylim)


ax2 = plt.subplot(gs[1])
plt.plot(Energy, Mass, 'k')
plt.ylabel('Phase (Mass Fraction)', labelpad = 5)
plt.xlabel('Energy Added (kJ)')
plt.xlim(left = minE, right = maxE)
plt.ylim(bottom = -0.2, top = 2.2)
plt.xticks(xticks)
plt.yticks([0,1,2], ['Solid', 'Liquid', 'Gas'], rotation = 90, va = 'center')
ax2.get_yaxis().set_label_coords(-0.065,0.5)

ylim = ax2.get_ylim()
xlim = ax2.get_xlim()
section_plot(xlim, ylim)

plt.subplots_adjust(hspace = None)
plt.tight_layout()
plt.show()
