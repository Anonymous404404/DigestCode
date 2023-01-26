import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def extract_data(loss_data):
    l = []
    t = []
    c = []
    for i in range(len(loss_data)):
        if len(loss_data[i][0][0][0]) == 4:
            l.append([np.amin(loss_data[i][j][0], axis=1) for j in range(len(loss_data[0]))])
        else:
            l.append([loss_data[i][j][0] for j in range(len(loss_data[0]))])
        t.append([loss_data[i][j][1] for j in range(len(loss_data[0]))])
        c.append([loss_data[i][j][2] for j in range(len(loss_data[0]))])
    return l, t, c


def linear_estimate(data, t, ind):
    res = []
    for i in ind:
        for j in range(len(t)):
            if t[j] >= i:
                res.append(data[j - 1] + (i - t[j - 1]) / (t[j] - t[j - 1]) * (data[j] - data[j - 1]))
                break
    return res

#%%
with open("./result/simulation_result", "r") as fp:
    data = json.load(fp)
    # data = [[item] for item in data]
#%%
with open("./result/Digest_compare", "r") as fp:
    dataa =json.load(fp)
    # data = [[item] for item in data]
data += [dataa[0],dataa[2]]
#%%
with open("./result/Digest_multistream_differentH", "r") as fp:
    data+=json.load(fp)
    # data = [[item] for item in data]
#%%

iter_loss, iter_t, iter_comm = extract_data(data)
#%%
t_loss = []
t_comm = []
time = range(0, 10 ** 6, 1000)
for i in range(len(iter_loss)):
    print(i)
    t_loss.append([linear_estimate(iter_loss[i][j], iter_t[i][j], time) for j in range(len(iter_loss[0]))])
    t_comm.append([linear_estimate(iter_comm[i][j], iter_t[i][j], time) for j in range(len(iter_comm[0]))])
#%%

av_iter_loss = []
std_iter_loss = []
av_t_loss = []
std_t_loss = []
av_iter_comm = []
std_iter_comm = []
av_t_comm = []
std_t_comm = []
for i in range(len(iter_loss)):
    av_iter_loss.append(np.average(iter_loss[i], axis=0))
    std_iter_loss.append(np.std(iter_loss[i], axis=0))
    av_t_loss.append(np.average(t_loss[i], axis=0))
    std_t_loss.append(np.std(t_loss[i], axis=0))
    av_iter_comm.append(np.average(iter_comm[i], axis=0))
    std_iter_comm.append(np.std(iter_comm[i], axis=0))
    av_t_comm.append(np.average(t_comm[i], axis=0))
    std_t_comm.append(np.std(t_comm[i], axis=0))

av_iter_loss = np.array([av_iter_loss[i][:500] for i in range(len(av_iter_loss))])
std_iter_loss = np.array([std_iter_loss[i][:500] for i in range(len(std_iter_loss))])
av_iter_comm = np.array([av_iter_comm[i][:500] for i in range(len(av_iter_comm))])
std_iter_comm = np.array([std_iter_comm[i][:500] for i in range(len(std_iter_comm))])

av_t_loss = np.array([av_t_loss[i][:1000] for i in range(len(av_t_loss))])
std_t_loss = np.array([std_t_loss[i][:1000] for i in range(len(std_t_loss))])
av_t_comm = np.array([av_t_comm[i][:1000] for i in range(len(av_t_comm))])
std_t_comm = np.array([std_t_comm[i][:1000] for i in range(len(std_t_comm))])

#%%
fig,ax = plt.subplots(figsize=(12,10))
plt.rcParams.update({'font.size': 19})
plt.xticks(fontsize = 19)
plt.yticks(fontsize = 19)
colors = matplotlib.cm.tab20(range(20))
b=0
markers=["o","X","P","^","v","s","h","<",">","d","*"]
every=[50,51,52,53,54,55,56,57,58,59,60]
order = [10,0,9,5,5,6,5,6,6]

# y = range(0,len(av_iter_loss[0]))
# ax.plot(y,av_iter_loss[b:,0:].T)
# for i,line in enumerate(ax.get_lines()):
#     line.set_marker(markers[i])
#     line.set_markevery(every[i])
#     line.set_color(colors[i])
#     line.set_markersize(10)
# for i in range(b,len(av_iter_loss)):
#     plt.fill_between(y, av_iter_loss[i,0:].T - std_iter_loss[i,0:].T, av_iter_loss[i,0:].T + std_iter_loss[i,0:].T,
#                  color=colors[i], alpha=0.2)
# plt.xlim(0,500)
# # plt.ylim(0.13,0.2)
# plt.ylim(0.2,0.6)
# ax.set_ylabel('Global loss')
# ax.set_xlabel('Iteration(1e3)')

y = range(0,1000)
ax.plot(y,av_t_loss[:,0:].T)
for i,line in enumerate(ax.get_lines()):
    line.set_marker(markers[i])
    line.set_markevery(every[i])
    line.set_color(colors[i])
    line.set_markersize(10)
for i in range(len(av_t_loss)):
    plt.fill_between(y, av_t_loss[i,0:].T - 1*std_t_loss[i,0:].T, av_t_loss[i,0:].T + 1*std_t_loss[i,0:].T,
                 color=colors[i], alpha=0.2)
# plt.ylim(0.13,0.2)
plt.ylim(0.2,0.6)
ax.set_xlabel('Time/s')
ax.set_ylabel('Global loss')
plt.xlim(0,1000)


# y = range(0,len(av_iter_comm[0]))
# ax.plot(y,av_iter_comm[:,0:].T)
# for i,line in enumerate(ax.get_lines()):
#     line.set_marker(markers[i])
#     line.set_markevery(every[i])
#     line.set_color(colors[i])
#     line.set_markersize(10)
# for i in range(len(av_iter_comm)):
#     plt.fill_between(y, av_iter_comm[i,0:].T - std_iter_comm[i,0:].T, av_iter_comm[i,0:].T + std_iter_comm[i,0:].T,
#                  color=colors[i], alpha=0.2)
# plt.xlim(0,500)
# ax.set_ylabel('Global communication')
# ax.set_xlabel('Iteration(1e3)')
# ax.set_yscale('log')

# y = range(0,1000)
# ax.plot(y,av_t_comm[:,0:].T)
# for i,line in enumerate(ax.get_lines()):
#     line.set_marker(markers[i])
#     line.set_markevery(every[i])
#     line.set_color(colors[i])
#     line.set_markersize(10)
# for i in range(len(av_t_comm)):
#     plt.fill_between(y, av_t_comm[i,0:].T - 1*std_t_comm[i,0:].T, av_t_comm[i,0:].T + 1*std_t_comm[i,0:].T,
#                  color=colors[i], alpha=0.2)
# plt.xlim(0,1000)
# ax.set_ylabel('Global communication')
# ax.set_xlabel('Time/s')
# ax.set_yscale('log')

#"Sync-Digest / H=800","Sync-Digest-MAB / H=800"
# ax.legend(["Sync-Gossip / H=1"
#            ,"One link-Gossip","Uniform Random Walk",
#            "5 Multi-Gossip-PGA / P=5","Real Avg-Gossip-PGA / P=5","Async-Gossip","Sync-Gossip / H=200","DIGEST / H=200"],ncol=2)#,"f *"])

# ax.legend(["DIGEST / R=1","DIGEST / R=5","DIGEST / R=10"
#           ,"DIGEST / R=20","DIGEST / R=30","Async-Gossip"],ncol=2)#,"f *"])

# ax.legend(["DIGEST-multi-stream / H=20","DIGEST-multi-stream / H=40","DIGEST-multi-stream / H=80","DIGEST-multi-stream / H=100"],ncol=2)#,"f *"])

ax.legend(["DIGEST-single-stream/ H=800", "DIGEST-multi-stream/ H=800","DIGEST-multi-stream / H=400","DIGEST-multi-stream / H=100",
           "DIGEST-multi-stream / H=20","Async-Gossip"],ncol=1)#,"f *"])

# ax.legend(["Fully connected","Line","Cluster"],ncol=2)#,"f *"])

ax.grid(True,which="both")
# ax.legend(["Decentralized - No MAB - H=1e3","Decentralized - No MAB - H=1e5","Decentralized","Gossip - One link allowed","Uniform Random Walk","f *"])
plt.savefig("./figures/l_t_multist.pdf",dpi =600,bbox_inches='tight',format='pdf')
# # files.download("line-non-IID-s.png")