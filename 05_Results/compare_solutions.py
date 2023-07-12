#!/usr/bin/env python
# coding: utf-8

# In[143]:


from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[144]:


get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.prototype._should_scroll = function(lines) {\n    return false;\n}')


# In[145]:


import matplotlib 

matplotlib.rc('xtick', labelsize=25) 
matplotlib.rc('ytick', labelsize=25)

font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 25}

matplotlib.rc('font', **font)


# In[179]:


import numpy as np
import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import interpolate # https://mljar.com/blog/matplotlib-colors/
import matplotlib.colors as mcolors
import matplotlib.patches as mpatch


# In[204]:


#print(mcolors.CSS4_COLORS)
#print(mcolors.TABLEAU_COLORS)
#print(mcolors.XKCD_COLORS)


# In[214]:


overlap = {name for name in mcolors.CSS4_COLORS if f'xkcd:{name}' in mcolors.XKCD_COLORS}
for j, color_name in enumerate(sorted(overlap)):
    print(j, color_name)
    css4 = mcolors.CSS4_COLORS[color_name]
    xkcd = mcolors.XKCD_COLORS[f'xkcd:{color_name}'].upper()
    rgba = mcolors.to_rgba_array([css4, xkcd])
    luma = 0.299 * rgba[:, 0] + 0.587 * rgba[:, 1] + 0.114 * rgba[:, 2]
    css4_text_color = 'k' if luma[0] > 0.5 else 'w'
    xkcd_text_color = 'k' if luma[1] > 0.5 else 'w'


# In[184]:


# compute average pressure from seven pressure probes
y, p = np.loadtxt('../04_Simulation/VT_NASA_BeVERLI_3D_Hill_Baseline_RANS_k_Omega_SST_45Deg_Mesh_VT_11M_Re_250k/postProcessing/sampleDict/110000//seven_probes_loc_ave_p_ref_cal_k_p.xy', usecols=(0, 2), unpack=True)

fig = plt.figure(figsize=(20,15))
plt.scatter(y,p, label='pressure from 7 probes', c='#acc2d9')
plt.axhline(y=p.mean(), label='Ave. pressure')
plt.xlabel('y/H', fontsize=20)
plt.ylabel('Cp',  fontsize=20)
plt.legend()
plt.grid()
plt.show()
plt.close()

print(p)
print('Average Reference Pressure =', p.mean())


# In[148]:


# Freestream values (required to compute Cp)
pref   = p.mean() 
rhoinf = 1.0 
uinf   = 1.0


# In[149]:


# Load VT experimental data
Cp_vs_x_H = pd.read_csv("../02_Data/Cp_vs_x_H.csv")
Cp_vs_z_H = pd.read_csv("../02_Data/Cp_vs_z_H.csv")


# In[150]:


# 01. VT_NASA_BeVERLI_3D_Hill_Baseline_RANS_k_Omega_SST_45Deg_Mesh_VT_Set3L4_Re_250k
# 02. VT_NASA_BeVERLI_3D_Hill_Baseline_RANS_k_Omega_SST_45Deg_Mesh_VT_11M_Re_250k
# 03. VT_NASA_BeVERLI_3D_Hill_GEP______RANS_k_Omega_SST_45Deg_Mesh_VT_Set3L4_Re_250k
# 04. VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_ON
# 05. VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_OFF
# 06. VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_OFF_Rij_ON
# 07. VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_OFF_Re_250k
# 08. BUMP_Re_250k_Set3L4_45deg_BSL_no_mapping
# 09. BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_LM-L2-MC  (SC)
# 10. BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_LM-L2-MC  (SC)
# 11. BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_LM-L2-MC   (SC)
# 12. BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_EVE-MO-52 (PH)
# 13. BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE-MO-52  (PH)
# 14. BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_EVE-MO-52 (PH)
# 15. BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON
# 16. BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF
# 17. BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON
# 18. BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_OFF
# 19. BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE-MO-52


# In[151]:


base = '../04_Simulation/' # 01
testcase = 'VT_NASA_BeVERLI_3D_Hill_Baseline_RANS_k_Omega_SST_45Deg_Mesh_VT_Set3L4_Re_250k/'
df_BSL_Set3L4_FloorSliceX_0 = pd.read_csv(base+testcase+'FloorSliceX_0.csv')
df_BSL_Set3L4_FloorSliceZ_0 = pd.read_csv(base+testcase+'FloorSliceZ_0.csv')
VT_NASA_BeVERLI_3D_Hill_Baseline_RANS_k_Omega_SST_45Deg_Mesh_VT_Set3L4_Re_250k_FloorSliceX_0 = pd.read_csv(base+testcase+'FloorSliceX_0.csv')
VT_NASA_BeVERLI_3D_Hill_Baseline_RANS_k_Omega_SST_45Deg_Mesh_VT_Set3L4_Re_250k_FloorSliceZ_0 = pd.read_csv(base+testcase+'FloorSliceZ_0.csv')


# In[152]:


base = '../04_Simulation/' # 02
testcase = 'VT_NASA_BeVERLI_3D_Hill_Baseline_RANS_k_Omega_SST_45Deg_Mesh_VT_11M_Re_250k/'
df_BSL_11M_FloorSliceX_0 = pd.read_csv(base+testcase+'FloorSliceX_0.csv')
df_BSL_11M_FloorSliceZ_0 = pd.read_csv(base+testcase+'FloorSliceZ_0.csv')
VT_NASA_BeVERLI_3D_Hill_Baseline_RANS_k_Omega_SST_45Deg_Mesh_VT_11M_Re_250k_FloorSliceX_0 = pd.read_csv(base+testcase+'FloorSliceX_0.csv')
VT_NASA_BeVERLI_3D_Hill_Baseline_RANS_k_Omega_SST_45Deg_Mesh_VT_11M_Re_250k_FloorSliceZ_0 = pd.read_csv(base+testcase+'FloorSliceZ_0.csv')


# In[153]:


base = '../04_Simulation/' #03
testcase = 'VT_NASA_BeVERLI_3D_Hill_GEP______RANS_k_Omega_SST_45Deg_Mesh_VT_Set3L4_Re_250k/'
df_GEP_Set3L4_FloorSliceX_0 = pd.read_csv(base+testcase+'FloorSliceX_0.csv')
df_GEP_Set3L4_FloorSliceZ_0 = pd.read_csv(base+testcase+'FloorSliceZ_0.csv')
VT_NASA_BeVERLI_3D_Hill_GEP______RANS_k_Omega_SST_45Deg_Mesh_VT_Set3L4_Re_250k_FloorSliceX_0 = pd.read_csv(base+testcase+'FloorSliceX_0.csv')
VT_NASA_BeVERLI_3D_Hill_GEP______RANS_k_Omega_SST_45Deg_Mesh_VT_Set3L4_Re_250k_FloorSliceZ_0 = pd.read_csv(base+testcase+'FloorSliceZ_0.csv')


# In[154]:


base = '../04_Simulation/' # 04
testcase = 'VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_ON/'
df_GEP_aij_ON_Rij_ON_11M_FloorSliceX_0 = pd.read_csv(base+testcase+'FloorSliceX_0.csv')
df_GEP_aij_ON_Rij_ON_11M_FloorSliceZ_0 = pd.read_csv(base+testcase+'FloorSliceZ_0.csv')
VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_ON_FloorSliceX_0 = pd.read_csv(base+testcase+'FloorSliceX_0.csv')
VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_ON_FloorSliceZ_0 = pd.read_csv(base+testcase+'FloorSliceZ_0.csv')


# In[155]:


base = '../04_Simulation/' # 05
testcase = 'VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_OFF/'
df_GEP_aij_ON_Rij_OFF_11M_FloorSliceX_0 = pd.read_csv(base+testcase+'FloorSliceX_0.csv')
df_GEP_aij_ON_Rij_OFF_11M_FloorSliceZ_0 = pd.read_csv(base+testcase+'/FloorSliceZ_0.csv')
VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_OFF_FloorSliceX_0 = pd.read_csv(base+testcase+'FloorSliceX_0.csv')
VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_OFF_FloorSliceZ_0 = pd.read_csv(base+testcase+'/FloorSliceZ_0.csv')


# In[156]:


base = '../04_Simulation/' # 06
testcase = 'VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_OFF_Rij_ON/'
df_GEP_aij_OFF_Rij_ON_11M_FloorSliceX_0 = pd.read_csv(base+testcase+'FloorSliceX_0.csv')
df_GEP_aij_OFF_Rij_ON_11M_FloorSliceZ_0 = pd.read_csv(base+testcase+'FloorSliceZ_0.csv')
VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_OFF_Rij_ON_FloorSliceX_0 = pd.read_csv(base+testcase+'FloorSliceX_0.csv')
VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_OFF_Rij_ON_FloorSliceZ_0 = pd.read_csv(base+testcase+'FloorSliceZ_0.csv')


# In[157]:


base = '../04_Simulation/' # 07
testcase = 'VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_OFF_Re_250k/'
df_GEP_aij_ON_Rij_OFF_11M_run0_FloorSliceX_0 = pd.read_csv(base+testcase+'FloorSliceX_0.csv')
df_GEP_aij_ON_Rij_OFF_11M_run0_FloorSliceZ_0 = pd.read_csv(base+testcase+'FloorSliceZ_0.csv')
VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_OFF_Re_250k_FloorSliceX_0 = pd.read_csv(base+testcase+'FloorSliceX_0.csv')
VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_OFF_Re_250k_FloorSliceZ_0 = pd.read_csv(base+testcase+'FloorSliceZ_0.csv')


# In[158]:


base = '../04_Simulation/45deg/L4/BSL/' # 08
testcase = 'BUMP_Re_250k_Set3L4_45deg_BSL_no_mapping/'
df_BUMP_Re_250k_Set3L4_45deg_BSL_no_mapping_FloorSliceX_0 = pd.read_csv(base+testcase+'FloorSliceX_0.csv')
df_BUMP_Re_250k_Set3L4_45deg_BSL_no_mapping_FloorSliceZ_0 = pd.read_csv(base+testcase+'FloorSliceZ_0.csv')
BUMP_Re_250k_Set3L4_45deg_BSL_no_mapping_FloorSliceX_0 = pd.read_csv(base+testcase+'FloorSliceX_0.csv')
BUMP_Re_250k_Set3L4_45deg_BSL_no_mapping_FloorSliceZ_0 = pd.read_csv(base+testcase+'FloorSliceZ_0.csv')


# In[159]:


base = '../04_Simulation/45deg/L4/GEP/Fabians_models/SC/'
# 09
df_GEP_Set3L4_FloorSliceX_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_LM_L2_MC = pd.read_csv(base+'BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_LM-L2-MC/FloorSliceX_0.csv')
df_GEP_Set3L4_FloorSliceZ_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_LM_L2_MC = pd.read_csv(base+'BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_LM-L2-MC/FloorSliceZ_0.csv')
BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_LM_L2_MC_FloorSliceX_0 = pd.read_csv(base+'BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_LM-L2-MC/FloorSliceX_0.csv')
BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_LM_L2_MC_FloorSliceZ_0 = pd.read_csv(base+'BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_LM-L2-MC/FloorSliceZ_0.csv')
# 10
df_GEP_Set3L4_FloorSliceX_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_LM_L2_MC = pd.read_csv(base+'BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_LM-L2-MC/FloorSliceX_0.csv')
df_GEP_Set3L4_FloorSliceZ_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_LM_L2_MC = pd.read_csv(base+'BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_LM-L2-MC/FloorSliceZ_0.csv')
BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_LM_L2_MC_FloorSliceX_0 = pd.read_csv(base+'BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_LM-L2-MC/FloorSliceX_0.csv')
BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_LM_L2_MC_FloorSliceZ_0 = pd.read_csv(base+'BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_LM-L2-MC/FloorSliceZ_0.csv')
# 11
df_GEP_Set3L4_FloorSliceX_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_LM_L2_MC = pd.read_csv(base+'BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_LM-L2-MC/FloorSliceX_0.csv')
df_GEP_Set3L4_FloorSliceZ_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_LM_L2_MC = pd.read_csv(base+'BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_LM-L2-MC/FloorSliceZ_0.csv')
BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_LM_L2_MC_FloorSliceX_0 = pd.read_csv(base+'BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_LM-L2-MC/FloorSliceX_0.csv')
BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_LM_L2_MC_FloorSliceZ_0 = pd.read_csv(base+'BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_LM-L2-MC/FloorSliceZ_0.csv')


# In[160]:


base = '../04_Simulation/45deg/L4/GEP/Fabians_models/PH/'
# 12
df_GEP_Set3L4_FloorSliceX_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_EVE_MO_52 = pd.read_csv(base+'BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_EVE-MO-52/FloorSliceX_0.csv')
df_GEP_Set3L4_FloorSliceZ_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_EVE_MO_52 = pd.read_csv(base+'BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_EVE-MO-52/FloorSliceZ_0.csv')
BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_EVE_MO_52_FloorSliceX_0 = pd.read_csv(base+'BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_EVE-MO-52/FloorSliceX_0.csv')
BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_EVE_MO_52_FloorSliceZ_0 = pd.read_csv(base+'BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_EVE-MO-52/FloorSliceZ_0.csv')
# 13
df_GEP_Set3L4_FloorSliceX_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE_MO_52 = pd.read_csv(base+'BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE-MO-52/FloorSliceX_0.csv')
df_GEP_Set3L4_FloorSliceZ_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE_MO_52 = pd.read_csv(base+'BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE-MO-52/FloorSliceZ_0.csv')
BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE_MO_52_FloorSliceX_0 = pd.read_csv(base+'BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE-MO-52/FloorSliceX_0.csv')
BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE_MO_52_FloorSliceZ_0 = pd.read_csv(base+'BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE-MO-52/FloorSliceZ_0.csv')
# 14
df_GEP_Set3L4_FloorSliceX_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_EVE_MO_52 = pd.read_csv(base+'BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_EVE-MO-52/FloorSliceX_0.csv')
df_GEP_Set3L4_FloorSliceZ_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_EVE_MO_52 = pd.read_csv(base+'BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_EVE-MO-52/FloorSliceZ_0.csv')
BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_EVE_MO_52_FloorSliceX_0 = pd.read_csv(base+'BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_EVE-MO-52/FloorSliceX_0.csv')
BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_EVE_MO_52_FloorSliceZ_0 = pd.read_csv(base+'BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_EVE-MO-52/FloorSliceZ_0.csv')


# In[161]:


base     = '../04_Simulation/45deg/L4/GEP/' # 15
testcase = 'BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON/'
df_GEP_Set3L4_FloorSliceX_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON = pd.read_csv(base+testcase+'FloorSliceX_0.csv')
df_GEP_Set3L4_FloorSliceZ_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON = pd.read_csv(base+testcase+'FloorSliceZ_0.csv')
BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_FloorSliceX_0 = pd.read_csv(base+testcase+'FloorSliceX_0.csv')
BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_FloorSliceZ_0 = pd.read_csv(base+testcase+'FloorSliceZ_0.csv')

base     = '../04_Simulation/45deg/L4/GEP/' # 16
testcase = 'BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF/'
df_GEP_Set3L4_FloorSliceX_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF = pd.read_csv(base+testcase+'/FloorSliceX_0.csv')
df_GEP_Set3L4_FloorSliceZ_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF = pd.read_csv(base+testcase+'FloorSliceZ_0.csv')
BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_FloorSliceX_0 = pd.read_csv(base+testcase+'FloorSliceX_0.csv')
BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_FloorSliceZ_0 = pd.read_csv(base+testcase+'FloorSliceZ_0.csv')

base     = '../04_Simulation/45deg/L4/GEP/' # 17
testcase = 'BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON/'
df_GEP_Set3L4_FloorSliceX_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON = pd.read_csv(base+testcase+'FloorSliceX_0.csv')
df_GEP_Set3L4_FloorSliceZ_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON = pd.read_csv(base+testcase+'FloorSliceZ_0.csv')
BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_FloorSliceX_0 = pd.read_csv(base+testcase+'FloorSliceX_0.csv')
BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_FloorSliceZ_0 = pd.read_csv(base+testcase+'FloorSliceZ_0.csv')

base     = '../04_Simulation/45deg/L4/GEP/' # 18
testcase = 'BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_OFF/'
df_GEP_Set3L4_FloorSliceX_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_OFF = pd.read_csv(base+testcase+'FloorSliceX_0.csv')
df_GEP_Set3L4_FloorSliceZ_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_OFF = pd.read_csv(base+testcase+'FloorSliceZ_0.csv')
BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_OFF_FloorSliceX_0 = pd.read_csv(base+testcase+'FloorSliceX_0.csv')
BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_OFF_FloorSliceZ_0 = pd.read_csv(base+testcase+'FloorSliceZ_0.csv')


# In[162]:


base     = '../04_Simulation/45deg/L4/GEP/Fabians_models/blend_PH_SC/' # 19
testcase = 'BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE-MO-52/'
df_GEP_Set3L4_FloorSliceX_0_BUMP_Re_250k_Set3L4_45deg_GEP_blend_PH_SC = pd.read_csv(base+testcase+'FloorSliceX_0.csv')
df_GEP_Set3L4_FloorSliceZ_0_BUMP_Re_250k_Set3L4_45deg_GEP_blend_PH_SC = pd.read_csv(base+testcase+'FloorSliceZ_0.csv')
BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE_MO_52_FloorSliceX_0 = pd.read_csv(base+testcase+'FloorSliceX_0.csv')
BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE_MO_52_FloorSliceZ_0 = pd.read_csv(base+testcase+'FloorSliceZ_0.csv')


# In[200]:


#fig = plt.figure(figsize=(30,15))
#
#plt.scatter(Cp_vs_x_H['x/H'],                                                      Cp_vs_x_H['Cp'],                                                                              c = 'k',     label = 'VT Exp.', s = 100)
#
#plt.scatter(df_BSL_11M_FloorSliceZ_0['Points:0'],                                  (df_BSL_11M_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                  c='magenta', label='BSL_11M')
#
#plt.scatter(df_BSL_Set3L4_FloorSliceZ_0['Points:0'],                               (df_BSL_Set3L4_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                               c='green',   label='BSL_Set3L4')
#
#plt.scatter(df_GEP_Set3L4_FloorSliceZ_0['Points:0'],                               (df_GEP_Set3L4_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                               c='red',     label='GEP_Set3L4')
#
#plt.scatter(df_GEP_aij_ON_Rij_ON_11M_FloorSliceZ_0['Points:0'],                    (df_GEP_aij_ON_Rij_ON_11M_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                    c='blue',    label='GEP_aij_ON_Rij_ON_11M')
#
#plt.scatter(df_GEP_aij_ON_Rij_OFF_11M_FloorSliceZ_0['Points:0'],                   (df_GEP_aij_ON_Rij_OFF_11M_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                   c='orange',  label='GEP_aij_ON_Rij_OFF_11M')
#
#plt.scatter(df_GEP_aij_ON_Rij_OFF_11M_run0_FloorSliceZ_0['Points:0'],              (df_GEP_aij_ON_Rij_OFF_11M_run0_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),              c='pink',    label='GEP_aij_ON_Rij_OFF_11M_run0')
#
#plt.scatter(df_GEP_aij_OFF_Rij_ON_11M_FloorSliceZ_0['Points:0'],                   (df_GEP_aij_OFF_Rij_ON_11M_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                   c='yellow',  label='GEP_aij_OFF_Rij_ON_11M')
#
#plt.scatter(df_BUMP_Re_250k_Set3L4_45deg_BSL_no_mapping_FloorSliceZ_0['Points:0'], (df_BUMP_Re_250k_Set3L4_45deg_BSL_no_mapping_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf), c='brown',   label='BSL_no_mapping')
#
#plt.xlabel('x/H', fontsize=20)
#plt.ylabel('Cp',  fontsize=20)
#plt.xlim([-8, 8])
#plt.ylim([-0.9, 0.5])
#plt.grid()
#plt.legend(fontsize=20)
#plt.show()
#plt.close()


# In[201]:


#fig = plt.figure(figsize=(30,15))
#
#plt.scatter(Cp_vs_z_H['z/H'],                                                      Cp_vs_z_H['Cp'],                                                                              c = 'k',     label = 'VT Exp.', s = 100)
#
#plt.scatter(df_BSL_11M_FloorSliceX_0['Points:2'],                                  (df_BSL_11M_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                  c='magenta', label='BSL_11M')
#
#plt.scatter(df_BSL_Set3L4_FloorSliceX_0['Points:2'],                               (df_BSL_Set3L4_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                               c='green',   label='BSL_Set3L4')
#
#plt.scatter(df_GEP_Set3L4_FloorSliceX_0['Points:2'],                               (df_GEP_Set3L4_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                               c='red',     label='GEP_Set3L4')
#
#plt.scatter(df_GEP_aij_ON_Rij_ON_11M_FloorSliceX_0['Points:2'],                    (df_GEP_aij_ON_Rij_ON_11M_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                    c='blue',    label='GEP_aij_ON_Rij_ON_11M')
#
#plt.scatter(df_GEP_aij_ON_Rij_OFF_11M_FloorSliceX_0['Points:2'],                   (df_GEP_aij_ON_Rij_OFF_11M_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                   c='orange',  label='GEP_aij_ON_Rij_OFF_11M')
#
#plt.scatter(df_GEP_aij_ON_Rij_OFF_11M_run0_FloorSliceX_0['Points:2'],              (df_GEP_aij_ON_Rij_OFF_11M_run0_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),              c='pink',    label='GEP_aij_ON_Rij_OFF_11M_run0')
#
#plt.scatter(df_GEP_aij_OFF_Rij_ON_11M_FloorSliceX_0['Points:2'],                   (df_GEP_aij_OFF_Rij_ON_11M_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                   c='yellow',  label='GEP_aij_OFF_Rij_ON_11M')
#
#plt.scatter(df_BUMP_Re_250k_Set3L4_45deg_BSL_no_mapping_FloorSliceX_0['Points:2'], (df_BUMP_Re_250k_Set3L4_45deg_BSL_no_mapping_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf), c='brown',   label='BSL_no_mapping')
#
#plt.xlabel('z/H', fontsize=20)
#plt.ylabel('Cp',  fontsize=20)
#plt.xlim([-4, 4])
#plt.ylim([-1.8, 0])
#plt.grid()
#plt.legend(fontsize=20)
#plt.show()
#plt.close()


# # Plot $C_p$ along centerspan ($x = 0$) plane

# In[230]:


fig = plt.figure(figsize=(30,15))
# 00 Exp. data
plt.scatter(Cp_vs_z_H['z/H'],                                                                                                  Cp_vs_z_H['Cp'],                                                                                                                            c = 'k',                             label='VT Exp.', s=100)
# 01
plt.plot(-VT_NASA_BeVERLI_3D_Hill_Baseline_RANS_k_Omega_SST_45Deg_Mesh_VT_Set3L4_Re_250k_FloorSliceX_0['Points:2'],            (VT_NASA_BeVERLI_3D_Hill_Baseline_RANS_k_Omega_SST_45Deg_Mesh_VT_Set3L4_Re_250k_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),            c=mcolors.CSS4_COLORS['aqua'],       label='VT_NASA_BeVERLI_3D_Hill_Baseline_RANS_k_Omega_SST_45Deg_Mesh_VT_Set3L4_Re_250k_FloorSliceX_0')
# 02
plt.plot(-VT_NASA_BeVERLI_3D_Hill_Baseline_RANS_k_Omega_SST_45Deg_Mesh_VT_11M_Re_250k_FloorSliceX_0['Points:2'],               (VT_NASA_BeVERLI_3D_Hill_Baseline_RANS_k_Omega_SST_45Deg_Mesh_VT_11M_Re_250k_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),               c=mcolors.CSS4_COLORS['aquamarine'], label='VT_NASA_BeVERLI_3D_Hill_Baseline_RANS_k_Omega_SST_45Deg_Mesh_VT_11M_Re_250k_FloorSliceX_0')
# 03
plt.plot(VT_NASA_BeVERLI_3D_Hill_GEP______RANS_k_Omega_SST_45Deg_Mesh_VT_Set3L4_Re_250k_FloorSliceX_0['Points:2'],             (VT_NASA_BeVERLI_3D_Hill_GEP______RANS_k_Omega_SST_45Deg_Mesh_VT_Set3L4_Re_250k_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),            c=mcolors.CSS4_COLORS['turquoise'],  label='VT_NASA_BeVERLI_3D_Hill_GEP______RANS_k_Omega_SST_45Deg_Mesh_VT_Set3L4_Re_250k_FloorSliceX_0')
# 04
plt.plot(-VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_ON_FloorSliceX_0['Points:2'],          (VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_ON_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),          c=mcolors.CSS4_COLORS['blue'],       label='VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_ON_FloorSliceX_0')
# 05
plt.plot(-VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_OFF_FloorSliceX_0['Points:2'],         (VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_OFF_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),         c=mcolors.CSS4_COLORS['brown'],      label='VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_OFF_FloorSliceX_0')
# 06  
plt.plot(-VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_OFF_Rij_ON_FloorSliceX_0['Points:2'],         (VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_OFF_Rij_ON_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),         c=mcolors.CSS4_COLORS['chartreuse'], label='VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_OFF_Rij_ON_FloorSliceX_0')
# 07
plt.plot(-VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_OFF_Re_250k_FloorSliceX_0['Points:2'], (VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_OFF_Re_250k_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf), c=mcolors.CSS4_COLORS['coral'],      label='VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_OFF_Re_250k_FloorSliceX_0')
# 08
plt.plot(-BUMP_Re_250k_Set3L4_45deg_BSL_no_mapping_FloorSliceX_0['Points:2'],                                                  (BUMP_Re_250k_Set3L4_45deg_BSL_no_mapping_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                                  c=mcolors.CSS4_COLORS['cyan'],       label='BUMP_Re_250k_Set3L4_45deg_BSL_no_mapping_FloorSliceX_0')
# 09
plt.plot(-BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_LM_L2_MC_FloorSliceX_0['Points:2'],                                     (BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_LM_L2_MC_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                     c=mcolors.CSS4_COLORS['darkblue'],   label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_LM_L2_MC_FloorSliceX_0')
# 10
plt.plot(-BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_LM_L2_MC_FloorSliceX_0['Points:2'],                                     (BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_LM_L2_MC_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                     c=mcolors.CSS4_COLORS['darkgreen'],  label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_LM_L2_MC_FloorSliceX_0')
# 11
plt.plot(-BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_LM_L2_MC_FloorSliceX_0['Points:2'],                                      (BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_LM_L2_MC_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                      c=mcolors.CSS4_COLORS['fuchsia'],    label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_LM_L2_MC_FloorSliceX_0')
# 12
plt.plot(-BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_EVE_MO_52_FloorSliceX_0['Points:2'],                                    (BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_EVE_MO_52_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                    c=mcolors.CSS4_COLORS['goldenrod'],  label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_EVE_MO_52_FloorSliceX_0')
# 13
plt.plot(-BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE_MO_52_FloorSliceX_0['Points:2'],                                     (BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE_MO_52_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                     c=mcolors.CSS4_COLORS['green'],      label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE_MO_52_FloorSliceX_0')
# 14
plt.plot(-BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_EVE_MO_52_FloorSliceX_0['Points:2'],                                    (BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_EVE_MO_52_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                    c=mcolors.CSS4_COLORS['violet'],     label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_EVE_MO_52_FloorSliceX_0')
# 15
plt.plot(-BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_FloorSliceX_0['Points:2'],                                               (BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                               c=mcolors.CSS4_COLORS['salmon'],     label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_FloorSliceX_0')
# 16
plt.plot(-BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_FloorSliceX_0['Points:2'],                                              (BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                              c=mcolors.CSS4_COLORS['maroon'],     label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_FloorSliceX_0')
# 17
plt.plot(-BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_FloorSliceX_0['Points:2'],                                              (BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                              c=mcolors.CSS4_COLORS['olive'],      label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_FloorSliceX_0')
# 18
plt.plot(-BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_OFF_FloorSliceX_0['Points:2'],                                             (BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_OFF_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                             c=mcolors.CSS4_COLORS['teal'],       label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_OFF_FloorSliceX_0')
# 19
plt.plot(-BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE_MO_52_FloorSliceX_0['Points:2'],                                     (BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE_MO_52_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                     c=mcolors.CSS4_COLORS['red'],        label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE_MO_52_FloorSliceX_0')
#
plt.xlabel('$z/H$', fontsize=20)
plt.ylabel('$C_p$', fontsize=20)
plt.xlim([-4, 4])
plt.ylim([-1.8, 0])
plt.grid()
plt.legend(fontsize=8)
plt.savefig('BH_Cp_X.pdf')
plt.show()
plt.close()


# In[243]:


fig = plt.figure(figsize=(30,15))
# 00 Exp. data
plt.scatter(Cp_vs_z_H['z/H'],                                                                                                  Cp_vs_z_H['Cp'],                                                                                                                            c = 'k',                             label='VT Exp.', s=100)
# 01
plt.plot(-VT_NASA_BeVERLI_3D_Hill_Baseline_RANS_k_Omega_SST_45Deg_Mesh_VT_Set3L4_Re_250k_FloorSliceX_0['Points:2'],            (VT_NASA_BeVERLI_3D_Hill_Baseline_RANS_k_Omega_SST_45Deg_Mesh_VT_Set3L4_Re_250k_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),            c=mcolors.CSS4_COLORS['aqua'],       label='VT_NASA_BeVERLI_3D_Hill_Baseline_RANS_k_Omega_SST_45Deg_Mesh_VT_Set3L4_Re_250k_FloorSliceX_0', linewidth=5)
## 02
#plt.plot(-VT_NASA_BeVERLI_3D_Hill_Baseline_RANS_k_Omega_SST_45Deg_Mesh_VT_11M_Re_250k_FloorSliceX_0['Points:2'],               (VT_NASA_BeVERLI_3D_Hill_Baseline_RANS_k_Omega_SST_45Deg_Mesh_VT_11M_Re_250k_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),               c=mcolors.CSS4_COLORS['aquamarine'], label='VT_NASA_BeVERLI_3D_Hill_Baseline_RANS_k_Omega_SST_45Deg_Mesh_VT_11M_Re_250k_FloorSliceX_0')
## 03
#plt.plot(VT_NASA_BeVERLI_3D_Hill_GEP______RANS_k_Omega_SST_45Deg_Mesh_VT_Set3L4_Re_250k_FloorSliceX_0['Points:2'],             (VT_NASA_BeVERLI_3D_Hill_GEP______RANS_k_Omega_SST_45Deg_Mesh_VT_Set3L4_Re_250k_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),            c=mcolors.CSS4_COLORS['turquoise'],  label='VT_NASA_BeVERLI_3D_Hill_GEP______RANS_k_Omega_SST_45Deg_Mesh_VT_Set3L4_Re_250k_FloorSliceX_0')
## 04
#plt.plot(-VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_ON_FloorSliceX_0['Points:2'],          (VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_ON_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),          c=mcolors.CSS4_COLORS['blue'],       label='VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_ON_FloorSliceX_0')
## 05
#plt.plot(-VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_OFF_FloorSliceX_0['Points:2'],         (VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_OFF_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),         c=mcolors.CSS4_COLORS['brown'],      label='VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_OFF_FloorSliceX_0')
## 06  
#plt.plot(-VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_OFF_Rij_ON_FloorSliceX_0['Points:2'],         (VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_OFF_Rij_ON_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),         c=mcolors.CSS4_COLORS['chartreuse'], label='VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_OFF_Rij_ON_FloorSliceX_0')
## 07
#plt.plot(-VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_OFF_Re_250k_FloorSliceX_0['Points:2'], (VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_OFF_Re_250k_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf), c=mcolors.CSS4_COLORS['coral'],      label='VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_OFF_Re_250k_FloorSliceX_0')
## 08
#plt.plot(-BUMP_Re_250k_Set3L4_45deg_BSL_no_mapping_FloorSliceX_0['Points:2'],                                                  (BUMP_Re_250k_Set3L4_45deg_BSL_no_mapping_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                                  c=mcolors.CSS4_COLORS['cyan'],       label='BUMP_Re_250k_Set3L4_45deg_BSL_no_mapping_FloorSliceX_0')
## 09
#plt.plot(-BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_LM_L2_MC_FloorSliceX_0['Points:2'],                                     (BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_LM_L2_MC_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                     c=mcolors.CSS4_COLORS['darkblue'],   label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_LM_L2_MC_FloorSliceX_0')
## 10
#plt.plot(-BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_LM_L2_MC_FloorSliceX_0['Points:2'],                                     (BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_LM_L2_MC_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                     c=mcolors.CSS4_COLORS['darkgreen'],  label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_LM_L2_MC_FloorSliceX_0')
## 11
#plt.plot(-BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_LM_L2_MC_FloorSliceX_0['Points:2'],                                      (BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_LM_L2_MC_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                      c=mcolors.CSS4_COLORS['fuchsia'],    label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_LM_L2_MC_FloorSliceX_0')
## 12
#plt.plot(-BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_EVE_MO_52_FloorSliceX_0['Points:2'],                                    (BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_EVE_MO_52_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                    c=mcolors.CSS4_COLORS['goldenrod'],  label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_EVE_MO_52_FloorSliceX_0')
## 13
#plt.plot(-BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE_MO_52_FloorSliceX_0['Points:2'],                                     (BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE_MO_52_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                     c=mcolors.CSS4_COLORS['green'],      label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE_MO_52_FloorSliceX_0')
## 14
#plt.plot(-BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_EVE_MO_52_FloorSliceX_0['Points:2'],                                    (BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_EVE_MO_52_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                    c=mcolors.CSS4_COLORS['violet'],     label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_EVE_MO_52_FloorSliceX_0')
## 15
#plt.plot(-BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_FloorSliceX_0['Points:2'],                                               (BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                               c=mcolors.CSS4_COLORS['salmon'],     label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_FloorSliceX_0')
## 16
#plt.plot(-BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_FloorSliceX_0['Points:2'],                                              (BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                              c=mcolors.CSS4_COLORS['maroon'],     label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_FloorSliceX_0')
## 17
#plt.plot(-BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_FloorSliceX_0['Points:2'],                                              (BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                              c=mcolors.CSS4_COLORS['olive'],      label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_FloorSliceX_0')
## 18
#plt.plot(-BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_OFF_FloorSliceX_0['Points:2'],                                             (BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_OFF_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                             c=mcolors.CSS4_COLORS['teal'],       label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_OFF_FloorSliceX_0')
# 19
plt.plot(-BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE_MO_52_FloorSliceX_0['Points:2'],                                     (BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE_MO_52_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                     c=mcolors.CSS4_COLORS['red'],        label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE_MO_52_FloorSliceX_0', linewidth=5)
#
plt.xlabel('$z/H$', fontsize=20)
plt.ylabel('$C_p$', fontsize=20)
plt.xlim([-4, 4])
plt.ylim([-1.8, 0])
plt.grid()
plt.legend(fontsize=8)
plt.savefig('BH_Cp_X.pdf')
plt.show()
plt.close()


# In[231]:


#fig = plt.figure(figsize=(30,15))
##
#plt.scatter(Cp_vs_z_H['z/H'], Cp_vs_z_H['Cp'], s = 100, c = 'k', label = 'VT Exp.')
##
#plt.plot(-df_BSL_11M_FloorSliceX_0['Points:2'], (df_BSL_11M_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf), c='magenta', label='BSL_11M')
##
#plt.plot(-df_BSL_Set3L4_FloorSliceX_0['Points:2'], (df_BSL_Set3L4_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf), c='red', linestyle='--', label='kOmegaSST-BSL', linewidth=5) #BSL_Set3L4
##
#plt.plot(df_GEP_Set3L4_FloorSliceX_0['Points:2'], (df_GEP_Set3L4_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf), c='red', label='GEP_Set3L4')
##
#plt.scatter(-df_GEP_aij_ON_Rij_ON_11M_FloorSliceX_0['Points:2'], (df_GEP_aij_ON_Rij_ON_11M_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf), c='blue', label='GEP_aij_ON_Rij_ON_11M')
##
#plt.scatter(-df_GEP_aij_ON_Rij_OFF_11M_FloorSliceX_0['Points:2'], (df_GEP_aij_ON_Rij_OFF_11M_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf), c='orange', label='GEP_aij_ON_Rij_OFF_11M')
##
#plt.scatter(-df_GEP_aij_ON_Rij_OFF_11M_run0_FloorSliceX_0['Points:2'], (df_GEP_aij_ON_Rij_OFF_11M_run0_FloorSliceX_0['p']-pref)/(0.5*rhoinf*uinf*uinf), c='pink', label='GEP_aij_ON_Rij_OFF_11M_run0')
##
###plt.plot(-df_GEP_Set3L4_FloorSliceX_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_LM_L2_MC['Points:2'], (df_GEP_Set3L4_FloorSliceX_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_LM_L2_MC['p']-pref)/(0.5*rhoinf*uinf*uinf), c='red', label='GEP_aij_OFF_Rij_ON_LM_L2_MC')
###plt.plot(-df_GEP_Set3L4_FloorSliceX_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_LM_L2_MC['Points:2'], (df_GEP_Set3L4_FloorSliceX_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_LM_L2_MC['p']-pref)/(0.5*rhoinf*uinf*uinf), c='blue', label='GEP_aij_ON_Rij_OFF_LM_L2_MC')
###plt.plot(-df_GEP_Set3L4_FloorSliceX_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_LM_L2_MC['Points:2'], (df_GEP_Set3L4_FloorSliceX_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_LM_L2_MC['p']-pref)/(0.5*rhoinf*uinf*uinf), c='green', label='GEP_aij_ON_Rij_ON_LM_L2_MC')
##
#plt.scatter(-df_GEP_Set3L4_FloorSliceX_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_EVE_MO_52['Points:2'], (df_GEP_Set3L4_FloorSliceX_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_EVE_MO_52['p']-pref)/(0.5*rhoinf*uinf*uinf), c='yellow', label='GEP_aij_OFF_Rij_ON_EVE_MO_52')
###plt.plot(-df_GEP_Set3L4_FloorSliceX_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE_MO_52['Points:2'], (df_GEP_Set3L4_FloorSliceX_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE_MO_52['p']-pref)/(0.5*rhoinf*uinf*uinf), c='violet', label='GEP_aij_ON_Rij_ON_EVE_MO_52')
###plt.plot(-df_GEP_Set3L4_FloorSliceX_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_EVE_MO_52['Points:2'], (df_GEP_Set3L4_FloorSliceX_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_EVE_MO_52['p']-pref)/(0.5*rhoinf*uinf*uinf), c='orange', label='GEP_aij_ON_Rij_OFF_EVE_MO_52')
##
#plt.plot(-df_GEP_Set3L4_FloorSliceX_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON['Points:2'], (df_GEP_Set3L4_FloorSliceX_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON['p']-pref)/(0.5*rhoinf*uinf*uinf), c='grey', label='GEP_aij_ON_Rij_ON')
#plt.plot(-df_GEP_Set3L4_FloorSliceX_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF['Points:2'], (df_GEP_Set3L4_FloorSliceX_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF['p']-pref)/(0.5*rhoinf*uinf*uinf), c='lime', label='GEP_aij_ON_Rij_OFF')
#plt.plot(-df_GEP_Set3L4_FloorSliceX_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_OFF['Points:2'], (df_GEP_Set3L4_FloorSliceX_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_OFF['p']-pref)/(0.5*rhoinf*uinf*uinf), c='green', label='GEP_aij_OFF_Rij_OFF')
#plt.plot(-df_GEP_Set3L4_FloorSliceX_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON['Points:2'], (df_GEP_Set3L4_FloorSliceX_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON['p']-pref)/(0.5*rhoinf*uinf*uinf), c='blue', label='GEP_aij_OFF_Rij_ON')
##
#plt.plot(-df_GEP_Set3L4_FloorSliceX_0_BUMP_Re_250k_Set3L4_45deg_GEP_blend_PH_SC['Points:2'], (df_GEP_Set3L4_FloorSliceX_0_BUMP_Re_250k_Set3L4_45deg_GEP_blend_PH_SC['p']-pref)/(0.5*rhoinf*uinf*uinf), c='green', label='kOmegaSST-GEP_blend_PH_WMSC', linewidth=5)
##
#plt.xlabel('$z/H$', fontsize=20)
#plt.ylabel('$C_p$', fontsize=20)
#plt.xlim([-4, 4])
#plt.ylim([-1.8, 0])
#plt.grid()
#plt.legend(fontsize=20)
#plt.savefig('BH_Cp_X.pdf')
#plt.show()
#plt.close()


# # Plot $C_p$ along centerline ($z = 0$) plane

# In[238]:


fig = plt.figure(figsize=(30,15))
# 00 Exp. data
plt.scatter(Cp_vs_x_H['x/H'],                                                                                                    Cp_vs_x_H['Cp'],                                                                                                                            c = 'k',                             label='VT Exp.', s=100)
# 01
plt.scatter(VT_NASA_BeVERLI_3D_Hill_Baseline_RANS_k_Omega_SST_45Deg_Mesh_VT_Set3L4_Re_250k_FloorSliceZ_0['Points:0'],            (VT_NASA_BeVERLI_3D_Hill_Baseline_RANS_k_Omega_SST_45Deg_Mesh_VT_Set3L4_Re_250k_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),            c=mcolors.CSS4_COLORS['aqua'],       label='VT_NASA_BeVERLI_3D_Hill_Baseline_RANS_k_Omega_SST_45Deg_Mesh_VT_Set3L4_Re_250k_FloorSliceZ_0')
# 02
plt.scatter(VT_NASA_BeVERLI_3D_Hill_Baseline_RANS_k_Omega_SST_45Deg_Mesh_VT_11M_Re_250k_FloorSliceZ_0['Points:0'],               (VT_NASA_BeVERLI_3D_Hill_Baseline_RANS_k_Omega_SST_45Deg_Mesh_VT_11M_Re_250k_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),               c=mcolors.CSS4_COLORS['aquamarine'], label='VT_NASA_BeVERLI_3D_Hill_Baseline_RANS_k_Omega_SST_45Deg_Mesh_VT_11M_Re_250k_FloorSliceZ_0')
# 03
plt.scatter(VT_NASA_BeVERLI_3D_Hill_GEP______RANS_k_Omega_SST_45Deg_Mesh_VT_Set3L4_Re_250k_FloorSliceZ_0['Points:0'],            (VT_NASA_BeVERLI_3D_Hill_GEP______RANS_k_Omega_SST_45Deg_Mesh_VT_Set3L4_Re_250k_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),            c=mcolors.CSS4_COLORS['turquoise'],  label='VT_NASA_BeVERLI_3D_Hill_GEP______RANS_k_Omega_SST_45Deg_Mesh_VT_Set3L4_Re_250k_FloorSliceZ_0')
# 04
plt.scatter(VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_ON_FloorSliceZ_0['Points:0'],          (VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_ON_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),          c=mcolors.CSS4_COLORS['blue'],       label='VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_ON_FloorSliceZ_0')
# 05
plt.scatter(VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_OFF_FloorSliceZ_0['Points:0'],         (VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_OFF_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),         c=mcolors.CSS4_COLORS['brown'],      label='VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_OFF_FloorSliceZ_0')
# 06  
plt.scatter(VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_OFF_Rij_ON_FloorSliceZ_0['Points:0'],         (VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_OFF_Rij_ON_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),         c=mcolors.CSS4_COLORS['chartreuse'], label='VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_OFF_Rij_ON_FloorSliceZ_0')
# 07
plt.scatter(VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_OFF_Re_250k_FloorSliceZ_0['Points:0'], (VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_OFF_Re_250k_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf), c=mcolors.CSS4_COLORS['coral'],      label='VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_OFF_Re_250k_FloorSliceZ_0')
# 08
plt.scatter(BUMP_Re_250k_Set3L4_45deg_BSL_no_mapping_FloorSliceZ_0['Points:0'],                                                  (BUMP_Re_250k_Set3L4_45deg_BSL_no_mapping_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                                  c=mcolors.CSS4_COLORS['cyan'],       label='BUMP_Re_250k_Set3L4_45deg_BSL_no_mapping_FloorSliceZ_0')
# 09
plt.scatter(BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_LM_L2_MC_FloorSliceZ_0['Points:0'],                                     (BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_LM_L2_MC_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                     c=mcolors.CSS4_COLORS['darkblue'],   label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_LM_L2_MC_FloorSliceZ_0')
# 10
plt.scatter(BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_LM_L2_MC_FloorSliceZ_0['Points:0'],                                     (BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_LM_L2_MC_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                     c=mcolors.CSS4_COLORS['darkgreen'],  label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_LM_L2_MC_FloorSliceZ_0')
# 11
plt.scatter(BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_LM_L2_MC_FloorSliceZ_0['Points:0'],                                      (BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_LM_L2_MC_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                      c=mcolors.CSS4_COLORS['fuchsia'],    label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_LM_L2_MC_FloorSliceZ_0')
# 12
plt.scatter(BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_EVE_MO_52_FloorSliceZ_0['Points:0'],                                    (BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_EVE_MO_52_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                    c=mcolors.CSS4_COLORS['goldenrod'],  label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_EVE_MO_52_FloorSliceZ_0')
# 13
plt.scatter(BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE_MO_52_FloorSliceZ_0['Points:0'],                                     (BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE_MO_52_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                     c=mcolors.CSS4_COLORS['green'],      label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE_MO_52_FloorSliceZ_0')
# 14
plt.scatter(BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_EVE_MO_52_FloorSliceZ_0['Points:0'],                                    (BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_EVE_MO_52_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                    c=mcolors.CSS4_COLORS['violet'],     label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_EVE_MO_52_FloorSliceZ_0')
# 15
plt.scatter(BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_FloorSliceZ_0['Points:0'],                                               (BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                               c=mcolors.CSS4_COLORS['salmon'],     label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_FloorSliceZ_0')
# 16
plt.scatter(BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_FloorSliceZ_0['Points:0'],                                              (BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                              c=mcolors.CSS4_COLORS['maroon'],     label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_FloorSliceZ_0')
# 17
plt.scatter(BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_FloorSliceZ_0['Points:0'],                                              (BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                              c=mcolors.CSS4_COLORS['olive'],      label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_FloorSliceZ_0')
# 18
plt.scatter(BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_OFF_FloorSliceZ_0['Points:0'],                                             (BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_OFF_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                             c=mcolors.CSS4_COLORS['teal'],       label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_OFF_FloorSliceZ_0')
# 19
plt.scatter(BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE_MO_52_FloorSliceZ_0['Points:0'],                                     (BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE_MO_52_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                     c=mcolors.CSS4_COLORS['red'],        label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE_MO_52_FloorSliceZ_0')
#
plt.xlabel('$x/H$', fontsize=20)
plt.ylabel('$C_p$', fontsize=20)
plt.xlim([-8, 8])
plt.ylim([-0.9, 0.6])
plt.xticks(np.arange(-8, 8, step=1.0))
plt.grid()
plt.legend(fontsize=8)
plt.savefig('BH_Cp_Z.pdf')
plt.show()
plt.close()


# In[241]:


fig = plt.figure(figsize=(30,15))
# 00 Exp. data
plt.scatter(Cp_vs_x_H['x/H'],                                                                                                    Cp_vs_x_H['Cp'],                                                                                                                            c = 'k',                             label='VT Exp.', s=100)
# 01
plt.scatter(VT_NASA_BeVERLI_3D_Hill_Baseline_RANS_k_Omega_SST_45Deg_Mesh_VT_Set3L4_Re_250k_FloorSliceZ_0['Points:0'],            (VT_NASA_BeVERLI_3D_Hill_Baseline_RANS_k_Omega_SST_45Deg_Mesh_VT_Set3L4_Re_250k_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),            c=mcolors.CSS4_COLORS['aqua'],       label='VT_NASA_BeVERLI_3D_Hill_Baseline_RANS_k_Omega_SST_45Deg_Mesh_VT_Set3L4_Re_250k_FloorSliceZ_0')
## 02
#plt.scatter(VT_NASA_BeVERLI_3D_Hill_Baseline_RANS_k_Omega_SST_45Deg_Mesh_VT_11M_Re_250k_FloorSliceZ_0['Points:0'],               (VT_NASA_BeVERLI_3D_Hill_Baseline_RANS_k_Omega_SST_45Deg_Mesh_VT_11M_Re_250k_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),               c=mcolors.CSS4_COLORS['aquamarine'], label='VT_NASA_BeVERLI_3D_Hill_Baseline_RANS_k_Omega_SST_45Deg_Mesh_VT_11M_Re_250k_FloorSliceZ_0')
## 03
#plt.scatter(VT_NASA_BeVERLI_3D_Hill_GEP______RANS_k_Omega_SST_45Deg_Mesh_VT_Set3L4_Re_250k_FloorSliceZ_0['Points:0'],            (VT_NASA_BeVERLI_3D_Hill_GEP______RANS_k_Omega_SST_45Deg_Mesh_VT_Set3L4_Re_250k_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),            c=mcolors.CSS4_COLORS['turquoise'],  label='VT_NASA_BeVERLI_3D_Hill_GEP______RANS_k_Omega_SST_45Deg_Mesh_VT_Set3L4_Re_250k_FloorSliceZ_0')
## 04
#plt.scatter(VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_ON_FloorSliceZ_0['Points:0'],          (VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_ON_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),          c=mcolors.CSS4_COLORS['blue'],       label='VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_ON_FloorSliceZ_0')
## 05
#plt.scatter(VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_OFF_FloorSliceZ_0['Points:0'],         (VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_OFF_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),         c=mcolors.CSS4_COLORS['brown'],      label='VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_OFF_FloorSliceZ_0')
## 06  
#plt.scatter(VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_OFF_Rij_ON_FloorSliceZ_0['Points:0'],         (VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_OFF_Rij_ON_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),         c=mcolors.CSS4_COLORS['chartreuse'], label='VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_OFF_Rij_ON_FloorSliceZ_0')
## 07
#plt.scatter(VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_OFF_Re_250k_FloorSliceZ_0['Points:0'], (VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_OFF_Re_250k_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf), c=mcolors.CSS4_COLORS['coral'],      label='VT_NASA_BeVERLI_3D_Hill_GEP_aijRij_hat_RANS_45Deg_Mesh_VT_11M_run0_aij_ON_Rij_OFF_Re_250k_FloorSliceZ_0')
## 08
#plt.scatter(BUMP_Re_250k_Set3L4_45deg_BSL_no_mapping_FloorSliceZ_0['Points:0'],                                                  (BUMP_Re_250k_Set3L4_45deg_BSL_no_mapping_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                                  c=mcolors.CSS4_COLORS['cyan'],       label='BUMP_Re_250k_Set3L4_45deg_BSL_no_mapping_FloorSliceZ_0')
## 09
plt.scatter(BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_LM_L2_MC_FloorSliceZ_0['Points:0'],                                     (BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_LM_L2_MC_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                     c=mcolors.CSS4_COLORS['darkblue'],   label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_LM_L2_MC_FloorSliceZ_0')
# 10
#plt.scatter(BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_LM_L2_MC_FloorSliceZ_0['Points:0'],                                     (BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_LM_L2_MC_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                     c=mcolors.CSS4_COLORS['darkgreen'],  label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_LM_L2_MC_FloorSliceZ_0')
## 11
#plt.scatter(BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_LM_L2_MC_FloorSliceZ_0['Points:0'],                                      (BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_LM_L2_MC_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                      c=mcolors.CSS4_COLORS['fuchsia'],    label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_LM_L2_MC_FloorSliceZ_0')
## 12
#plt.scatter(BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_EVE_MO_52_FloorSliceZ_0['Points:0'],                                    (BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_EVE_MO_52_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                    c=mcolors.CSS4_COLORS['goldenrod'],  label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_EVE_MO_52_FloorSliceZ_0')
## 13
#plt.scatter(BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE_MO_52_FloorSliceZ_0['Points:0'],                                     (BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE_MO_52_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                     c=mcolors.CSS4_COLORS['green'],      label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE_MO_52_FloorSliceZ_0')
## 14
#plt.scatter(BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_EVE_MO_52_FloorSliceZ_0['Points:0'],                                    (BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_EVE_MO_52_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                    c=mcolors.CSS4_COLORS['violet'],     label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_EVE_MO_52_FloorSliceZ_0')
## 15
#plt.scatter(BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_FloorSliceZ_0['Points:0'],                                               (BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                               c=mcolors.CSS4_COLORS['salmon'],     label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_FloorSliceZ_0')
## 16
#plt.scatter(BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_FloorSliceZ_0['Points:0'],                                              (BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                              c=mcolors.CSS4_COLORS['maroon'],     label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_FloorSliceZ_0')
## 17
#plt.scatter(BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_FloorSliceZ_0['Points:0'],                                              (BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                              c=mcolors.CSS4_COLORS['olive'],      label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_FloorSliceZ_0')
## 18
#plt.scatter(BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_OFF_FloorSliceZ_0['Points:0'],                                             (BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_OFF_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                             c=mcolors.CSS4_COLORS['teal'],       label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_OFF_FloorSliceZ_0')
# 19
plt.scatter(BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE_MO_52_FloorSliceZ_0['Points:0'],                                     (BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE_MO_52_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf),                                     c=mcolors.CSS4_COLORS['red'],        label='BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE_MO_52_FloorSliceZ_0')
#
plt.xlabel('$x/H$', fontsize=20)
plt.ylabel('$C_p$', fontsize=20)
plt.xlim([-8, 8])
plt.ylim([-0.9, 0.6])
plt.xticks(np.arange(-8, 8, step=1.0))
plt.grid()
plt.legend(fontsize=8)
plt.savefig('BH_Cp_Z.pdf')
plt.show()
plt.close()


# In[244]:


#fig = plt.figure(figsize=(30,15))
##
#plt.scatter(Cp_vs_x_H['x/H'], Cp_vs_x_H['Cp'], s = 100, c = 'k', label = 'VT Exp.')
##
#plt.scatter(df_BSL_11M_FloorSliceZ_0['Points:0'], (df_BSL_11M_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf), c='red', label='kOmegaSST-BSL') #BSL_11M
##
##plt.scatter(df_BSL_Set3L4_FloorSliceZ_0['Points:0'], (df_BSL_Set3L4_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf), c='green', label='BSL_Set3L4')
##
##plt.scatter(df_GEP_Set3L4_FloorSliceZ_0['Points:0'], (df_GEP_Set3L4_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf), c='red', label='GEP_Set3L4')
##
##plt.scatter(df_GEP_aij_ON_Rij_ON_11M_FloorSliceZ_0['Points:0'], (df_GEP_aij_ON_Rij_ON_11M_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf), c='blue', label='GEP_aij_ON_Rij_ON_11M')
##
##plt.scatter(df_GEP_aij_ON_Rij_OFF_11M_FloorSliceZ_0['Points:0'], (df_GEP_aij_ON_Rij_OFF_11M_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf), c='orange', label='GEP_aij_ON_Rij_OFF_11M')
##
##plt.scatter(df_GEP_aij_ON_Rij_OFF_11M_run0_FloorSliceZ_0['Points:0'], (df_GEP_aij_ON_Rij_OFF_11M_run0_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf), c='pink', label='GEP_aij_ON_Rij_OFF_11M_run0')
##
##plt.scatter(df_GEP_aij_OFF_Rij_ON_11M_FloorSliceZ_0['Points:0'], (df_GEP_aij_OFF_Rij_ON_11M_FloorSliceZ_0['p']-pref)/(0.5*rhoinf*uinf*uinf), c='yellow', label='GEP_aij_OFF_Rij_ON_11M')
#plt.scatter(df_GEP_Set3L4_FloorSliceZ_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_LM_L2_MC['Points:0'], (df_GEP_Set3L4_FloorSliceZ_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_LM_L2_MC['p']-pref)/(0.5*rhoinf*uinf*uinf), c='blue', label='kOmegaSST-GEP_aij_OFF_Rij_ON_LM_L2_MC')
##plt.scatter(df_GEP_Set3L4_FloorSliceZ_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_LM_L2_MC['Points:0'], (df_GEP_Set3L4_FloorSliceZ_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_LM_L2_MC['p']-pref)/(0.5*rhoinf*uinf*uinf), c='blue', label='GEP_aij_ON_Rij_OFF_LM_L2_MC')
##plt.scatter(df_GEP_Set3L4_FloorSliceZ_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_LM_L2_MC['Points:0'], (df_GEP_Set3L4_FloorSliceZ_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_LM_L2_MC['p']-pref)/(0.5*rhoinf*uinf*uinf), c='green', label='GEP_aij_ON_Rij_ON_LM_L2_MC')
##
##plt.scatter(df_GEP_Set3L4_FloorSliceZ_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_EVE_MO_52['Points:0'], (df_GEP_Set3L4_FloorSliceZ_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON_EVE_MO_52['p']-pref)/(0.5*rhoinf*uinf*uinf), c='yellow', label='GEP_aij_OFF_Rij_ON_EVE_MO_52')
##plt.scatter(df_GEP_Set3L4_FloorSliceZ_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE_MO_52['Points:0'], (df_GEP_Set3L4_FloorSliceZ_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON_EVE_MO_52['p']-pref)/(0.5*rhoinf*uinf*uinf), c='violet', label='GEP_aij_ON_Rij_ON_EVE_MO_52')
##plt.scatter(df_GEP_Set3L4_FloorSliceZ_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_EVE_MO_52['Points:0'], (df_GEP_Set3L4_FloorSliceZ_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF_EVE_MO_52['p']-pref)/(0.5*rhoinf*uinf*uinf), c='orange', label='GEP_aij_ON_Rij_OFF_EVE_MO_52')
##
##plt.plot(-df_GEP_Set3L4_FloorSliceZ_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON['Points:0'], (df_GEP_Set3L4_FloorSliceZ_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_ON['p']-pref)/(0.5*rhoinf*uinf*uinf), c='black', label='GEP_aij_ON_Rij_ON', marker='|')
##plt.plot(-df_GEP_Set3L4_FloorSliceZ_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF['Points:0'], (df_GEP_Set3L4_FloorSliceZ_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_ON_Rij_OFF['p']-pref)/(0.5*rhoinf*uinf*uinf), c='red', label='GEP_aij_ON_Rij_OFF', marker='|')
##plt.plot(-df_GEP_Set3L4_FloorSliceZ_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_OFF['Points:0'], (df_GEP_Set3L4_FloorSliceZ_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_OFF['p']-pref)/(0.5*rhoinf*uinf*uinf), c='green', label='GEP_aij_OFF_Rij_OFF', marker='|')
##plt.plot(-df_GEP_Set3L4_FloorSliceZ_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON['Points:0'], (df_GEP_Set3L4_FloorSliceZ_0_BUMP_Re_250k_Set3L4_45deg_GEP_aij_OFF_Rij_ON['p']-pref)/(0.5*rhoinf*uinf*uinf), c='blue', label='GEP_aij_OFF_Rij_ON', marker='|')
##
#plt.scatter(df_GEP_Set3L4_FloorSliceZ_0_BUMP_Re_250k_Set3L4_45deg_GEP_blend_PH_SC['Points:0'], (df_GEP_Set3L4_FloorSliceZ_0_BUMP_Re_250k_Set3L4_45deg_GEP_blend_PH_SC['p']-pref)/(0.5*rhoinf*uinf*uinf), c='green', label='kOmegaSST-GEP_blend_PH_WMSC')
##
#plt.xlabel('$x/H$', fontsize=20)
#plt.ylabel('$C_p$', fontsize=20)
#plt.xlim([-8, 8])
#plt.ylim([-0.9, 0.6])
#plt.xticks(np.arange(-8, 8, step=1.0))  # Set label locations.
#plt.grid()
#plt.legend(fontsize=20)
#plt.savefig('BH_Cp_Z.pdf')
#plt.show()
#plt.close()


# In[ ]:




