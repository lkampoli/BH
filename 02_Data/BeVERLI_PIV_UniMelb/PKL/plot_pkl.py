import pickle
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

with open('Plane11.pkl', 'rb') as f:
    data = pickle.load(f)
print(data)
#pd_data = pd.DataFrame.from_dict(data)
#print(pd_data)
#data.replace(np.nan,0)

X    = data["coordinates"]["X"].flatten()
Y    = data["coordinates"]["Y"].flatten()
U    = data["meanVel"]["U"].flatten()
V    = data["meanVel"]["V"].flatten()
W    = data["meanVel"]["W"].flatten()
UU   = data["reStress"]["UU"].flatten()
VV   = data["reStress"]["VV"].flatten()
WW   = data["reStress"]["WW"].flatten()
UV   = data["reStress"]["UV"].flatten()
UW   = data["reStress"]["UW"].flatten()
VW   = data["reStress"]["VW"].flatten()
dUdX = data["meanVelGrad"]["dUdX"].flatten()
dUdY = data["meanVelGrad"]["dUdY"].flatten()
dUdZ = data["meanVelGrad"]["dUdZ"].flatten()
dVdX = data["meanVelGrad"]["dVdX"].flatten()
dVdY = data["meanVelGrad"]["dVdY"].flatten()
dVdZ = data["meanVelGrad"]["dVdZ"].flatten()
dWdX = data["meanVelGrad"]["dWdX"].flatten()
dWdY = data["meanVelGrad"]["dWdY"].flatten()
dWdZ = data["meanVelGrad"]["dWdZ"].flatten()

print(U)

#plt.scatter(Y,U)
#plt.show()

#fig = plt.figure(figsize=(30,15))
#ax1 = fig.add_subplot(111)
#f = ax1.tricontourf(X, Y, U, levels=10)
##ax1.tricontour(f, case_1p0_refined['Points:0'], case_1p0_refined['Points:1'], (case_1p0_refined['TauDNS:3']-case_1p0_refined['Tau:3']), colors='k')
#ig.colorbar(f)
#ax1.set_xlabel('X')
#ax1.set_ylabel('Y')
#ax1.set_aspect(1.5)
#plt.title("U")
#plt.show()
#plt.close()
