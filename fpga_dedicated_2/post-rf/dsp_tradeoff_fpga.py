import matplotlib.pyplot as plt
import math

fname1="dsp_tradeoff_fpga.svg"

x=[140,225,450,580,835]
y=[1/2693036*200e6,1/1962220*200e6,1/785772*200e6,1/672479*200e6,1/407546*200e6]
acc=[75.95,77.24, 76.98,76.22,76.45]

jiang_co_design_x=[450]
jiang_co_design_fps=[35.5]
dnnbuilder_vgg_x=[680]
dnnbuilder_vgg_fps=[27.7]
dnnbuilder_vgg_acc=[70.48]
dnnbuilder_alexnet_x=[808]
dnnbuilder_alexnet_fps=[170]
dnnbuilder_alexnet_acc=[170]

our_vgg_x=[731]
our_vgg_fps=[104]
our_alexnet_x=[670]
our_alexnet_fps=[1/525443*200e6]
acc=sorted(acc)
fig, ax1 = plt.subplots()
ax1.set_xlabel('DSP')
ax1.set_ylabel('FPS')
ax1.plot(x, y, '-o',color="darkorange",fillstyle="none",linewidth=3,markersize=12,markeredgewidth=2)
ax1.tick_params(axis='y')
ax1.scatter(jiang_co_design_x,jiang_co_design_fps,s=120, facecolors='none', edgecolors='darkblue',linewidths=2)
ax1.scatter(our_vgg_x, our_vgg_fps,s=120, facecolors='none', edgecolors='slateblue',linewidths=2)
ax1.scatter(our_alexnet_x, our_alexnet_fps,s=120, facecolors='none', edgecolors='slateblue',linewidths=2)
# ax1.scatter(dnnbuilder_vgg_x,dnnbuilder_vgg_fps,s=120, facecolors='none', edgecolors='slateblue',linewidths=2)
# ax1.scatter(dnnbuilder_alexnet_x,dnnbuilder_alexnet_fps,s=120, facecolors='none', edgecolors='slateblue',linewidths=2)
ax1.set_ylim(0,600)
ax1.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Accuracy')  # we already handled the x-label with ax1
ax2.plot(x, acc,'--o',color="darkorange",fillstyle="none",linewidth=3,markersize=12,markeredgewidth=2)
ax2.scatter(our_vgg_x,dnnbuilder_vgg_acc,s=120, facecolors='none', edgecolors='slateblue',linestyle='dashed',linewidths=2)
ax2.set_ylim(67,82)
ax2.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax2.tick_params(axis='y')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(fname1)

