import matplotlib.pyplot as plt
import math
import numpy as np
import copy
fname1="plotter_space_vis.svg"
raw_data0=np.load('vis_data_noc0.npy',allow_pickle=True)
raw_data0=raw_data0.item()
# for i in raw_data0:
#     for j in range(len(raw_data0[i])):
#         raw_data0[i][j]=1/raw_data0[i][j]
#     raw_data0[i]=(np.array(raw_data0[i])-np.mean(raw_data0[i]))/np.std(raw_data0[i])
raw_data1=np.load('vis_data_noc1.npy',allow_pickle=True)
raw_data1=raw_data1.item()
# for i in raw_data1:
#     for j in range(len(raw_data1[i])):
#         raw_data1[i][j]=1/raw_data1[i][j]
#     raw_data1[i] = (np.array(raw_data1[i]) - np.mean(raw_data1[i])) / np.std(raw_data1[i])
raw_data2=np.load('vis_data_noc2.npy',allow_pickle=True)
raw_data2=raw_data2.item()
# for i in raw_data2:
#     for j in range(len(raw_data2[i])):
#         raw_data2[i][j]=1/raw_data2[i][j]
#     raw_data2[i] = (np.array(raw_data2[i]) - np.mean(raw_data2[i])) / np.std(raw_data2[i])
raw_data3=np.load('vis_data_noc3.npy',allow_pickle=True)
raw_data3=raw_data3.item()
# for i in raw_data3:
#     for j in range(len(raw_data3[i])):
#         raw_data3[i][j]=1/raw_data3[i][j]
#     raw_data3[i] = (np.array(raw_data3[i]) - np.mean(raw_data3[i])) / np.std(raw_data3[i])

fig, ax1 = plt.subplots()
ax1.set_xlabel('Generated Hardware Designs')
ax1.set_ylabel('Latency')
ax1.scatter(list(range(len(raw_data3['latency']))),[1/4/i for i in raw_data3['latency']],s=10, c='g')
ax1.scatter(list(range(len(raw_data2['latency']))),[1/4/i for i in raw_data2['latency']],s=10, c='red')
ax1.scatter(list(range(len(raw_data1['latency']))),[1/4/i for i in raw_data1['latency']],s=10, c='darkorange')
ax1.scatter(list(range(len(raw_data0['latency']))),[1/4/i for i in raw_data0['latency']],s=10, c='darkblue')
plt.savefig(fname1)

fname2="plotter_space_vis_ene.svg"
fig1, ax2 = plt.subplots()
ax2.set_xlabel('Generated Hardware Designs')
ax2.set_ylabel('Energy')
ax2.scatter(list(range(len(raw_data3['energy']))),[1/i*1000 for i in raw_data3['energy']],s=10, c='g')
ax2.scatter(list(range(len(raw_data2['energy']))),[1/i*1000 for i in raw_data2['energy']],s=10, c='red')
ax2.scatter(list(range(len(raw_data1['energy']))),[1/i*1000 for i in raw_data1['energy']],s=10, c='darkorange')
ax2.scatter(list(range(len(raw_data0['energy']))),[1/i*1000 for i in raw_data0['energy']],s=10, c='darkblue')
plt.savefig(fname2)


# fname3="plotter_space_vis_edp.svg"
# fig2, ax3 = plt.subplots()
# ax3.set_xlabel('Generated Hardware Designs')
# ax3.set_ylabel('EDP')
# ax3.scatter(list(range(len(raw_data3['edp']))),[1/i for i in raw_data3['edp']],s=10, c='g')
# ax3.scatter(list(range(len(raw_data2['edp']))),[1/i for i in raw_data2['edp']],s=10, c='red')
# ax3.scatter(list(range(len(raw_data1['edp']))),[1/i for i in raw_data1['edp']],s=10, c='darkorange')
# ax3.scatter(list(range(len(raw_data0['edp']))),[1/i for i in raw_data0['edp']],s=10, c='darkblue')
# plt.savefig(fname3)



fname4="plotter_space_vis_lat_ene.svg"
fig3, ax4 = plt.subplots()
ax4.set_xlabel('Energy')
ax4.set_ylabel('Latency')
ax4.scatter([i/1000 for i in raw_data3['energy']],[i*4 for i in raw_data3['latency']],s=10, c='g')
ax4.scatter([i/1000 for i in raw_data2['energy']],[i*4 for i in raw_data2['latency']],s=10, c='red')
ax4.scatter([i/1000 for i in raw_data1['energy']],[i*4 for i in raw_data1['latency']],s=10, c='darkorange')
ax4.scatter([i/1000 for i in raw_data0['energy']],[i*4 for i in raw_data0['latency']],s=10, c='darkblue')
ax4.set_xlim(0.3,1)
ax4.set_ylim(600,4000)
plt.savefig(fname4)


fname5="plotter_space_vis_lat_3d.svg"
fig4 = plt.figure()
ax5 = fig4.add_subplot(111, projection='3d')
# ax5.set_xlabel('Generated Hardware Designs')
# ax5.set_ylabel('NoC Designs')
# ax5.set_zlabel('Latency')
print(len(np.ones((len(raw_data3['latency']),))*3))
print(len(list(range(len(raw_data3['latency'])))))
ax5.scatter(list(range(len(raw_data3['latency']))),np.ones((len(raw_data3['latency']),))*3,[1/4/i*10e3 for i in raw_data3['latency']],s=10, c='g')
ax5.scatter(list(range(len(raw_data2['latency']))),np.ones((len(raw_data2['latency']),))*2,[1/4/i*10e3 for i in raw_data2['latency']],s=10, c='red')
ax5.scatter(list(range(len(raw_data1['latency']))),np.ones((len(raw_data1['latency']),)),[1/4/i*10e3 for i in raw_data1['latency']],s=10, c='darkorange')
ax5.scatter(list(range(len(raw_data0['latency']))),np.zeros((len(raw_data0['latency']),)),[1/4/i*10e3 for i in raw_data0['latency']],s=10, c='darkblue')
ax5.ticklabel_format(style='sci',axis='z')
ax5.zaxis.set_tick_params(labelsize=14)
ax5.yaxis.set_tick_params(labelsize=14)
ax5.xaxis.set_tick_params(labelsize=14)
plt.show()
plt.savefig(fname5)




fname6="plotter_space_vis_ene_3d.svg"
fig5 = plt.figure()
ax6 = fig5.add_subplot(111, projection='3d')
# ax5.set_xlabel('Generated Hardware Designs')
# ax5.set_ylabel('NoC Designs')
# ax5.set_zlabel('Latency')
print(len(np.ones((len(raw_data3['energy']),))*3))
print(len(list(range(len(raw_data3['energy'])))))
ax6.scatter(list(range(len(raw_data3['energy']))),np.ones((len(raw_data3['energy']),))*3,[1/i*1000 for i in raw_data3['energy']],s=10, c='g')
ax6.scatter(list(range(len(raw_data2['energy']))),np.ones((len(raw_data2['energy']),))*2,[1/i*1000 for i in raw_data2['energy']],s=10, c='red')
ax6.scatter(list(range(len(raw_data1['energy']))),np.ones((len(raw_data1['energy']),)),[1/i*1000 for i in raw_data1['energy']],s=10, c='darkorange')
ax6.scatter(list(range(len(raw_data0['energy']))),np.zeros((len(raw_data0['energy']),)),[1/i*1000 for i in raw_data0['energy']],s=10, c='darkblue')
ax6.ticklabel_format(style='sci',axis='z')
ax6.zaxis.set_tick_params(labelsize=14)
ax6.yaxis.set_tick_params(labelsize=14)
ax6.xaxis.set_tick_params(labelsize=14)
plt.show()
plt.savefig(fname5)



xlim=(0.3e3,1e3)
ylim=(600/4,4000/4)
raw_data0_cp= {'latency':[],'energy':[]}
raw_data1_cp= {'latency':[],'energy':[]}
raw_data2_cp= {'latency':[],'energy':[]}
raw_data3_cp= {'latency':[],'energy':[]}

for i in range(len(raw_data0['energy'])):
    if raw_data0['energy'][i]>xlim[0] and raw_data0['energy'][i]<xlim[1]  and raw_data0['latency'][i]>ylim[0] and raw_data0['latency'][i]<ylim[1]:
        raw_data0_cp['latency'].append(raw_data0['latency'][i])
        raw_data0_cp['energy'] .append( raw_data0['energy'][i])
for i in range(len(raw_data1['energy'])):
    if raw_data1['energy'][i]>xlim[0] and raw_data1['energy'][i]<xlim[1]  and raw_data1['latency'][i]>ylim[0] and raw_data1['latency'][i]<ylim[1]:
        raw_data1_cp['latency'].append(raw_data1['latency'][i])
        raw_data1_cp['energy'] .append( raw_data1['energy'][i])
for i in range(len(raw_data2['energy'])):
    if raw_data2['energy'][i]>xlim[0] and raw_data2['energy'][i]<xlim[1]  and raw_data2['latency'][i]>ylim[0] and raw_data2['latency'][i]<ylim[1]:
        raw_data2_cp['latency'].append(raw_data2['latency'][i])
        raw_data2_cp['energy'] .append(raw_data2['energy'][i])
for i in range(len(raw_data3['energy'])):
    if raw_data3['energy'][i]>xlim[0] and raw_data3['energy'][i]<xlim[1]  and raw_data3['latency'][i]>ylim[0] and raw_data3['latency'][i]<ylim[1]:
        raw_data3_cp['latency'].append(raw_data3['latency'][i])
        raw_data3_cp['energy'] .append( raw_data3['energy'][i])

fname7="plotter_space_vis_lat_ene_3d.svg"
fig6 = plt.figure()
ax7= fig6.add_subplot(111, projection='3d')

ax7.scatter([i/1000 for i in raw_data3_cp['energy']],[i*4 for i in raw_data3_cp['latency']],np.ones((len(raw_data3_cp['energy']),))*3,s=10, c='g')
ax7.set_xlim3d(0.3,1)
ax7.set_ylim3d(600,4000)
ax7.scatter([i/1000 for i in raw_data2_cp['energy']],[i*4 for i in raw_data2_cp['latency']],np.ones((len(raw_data2_cp['energy']),))*2,s=10, c='red')
ax7.scatter([i/1000 for i in raw_data1_cp['energy']],[i*4 for i in raw_data1_cp['latency']],np.ones((len(raw_data1_cp['energy']),))*1,s=10, c='darkorange')
ax7.scatter([i/1000 for i in raw_data0_cp['energy']],[i*4 for i in raw_data0_cp['latency']],np.ones((len(raw_data0_cp['energy']),))*0,s=10, c='darkblue')

ax7.ticklabel_format(style='sci',axis='z')
ax7.zaxis.set_tick_params(labelsize=14)
ax7.yaxis.set_tick_params(labelsize=14)
ax7.xaxis.set_tick_params(labelsize=14)
plt.show()
plt.savefig(fname7)
