# DIGEST

Fast and Communication Efficient Decentralized Learning with Local Updates

Abstract: Decentralized learning advocates the elimina-
tion of centralized parameter servers (aggregation
points) for potentially better utilization of under-
lying resources, delay reduction, and resiliency
against parameter server unavailability and catas-
trophic failures. Gossip based decentralized al-
gorithms, where each node in a network has its
own locally kept model on which it effectuates
the learning by talking to its neighbors, received
a lot of attention recently. Despite their potential,
Gossip algorithms introduce huge communication
costs. In this work, we show that nodes do not
need to communicate as frequently as in Gossip
for fast convergence; in fact, a sporadic exchange
of a global model is sufficient. Thus, we design
a fast and communication-efficient decentralized
learning mechanism; DIGEST by particularly fo-
cusing on stochastic gradient descent (SGD). DI-
GEST is a decentralized algorithm building on
local-SGD algorithms, which are originally de-
signed for communication efficient centralized
learning. We show through analysis and experi-
ments that DIGEST significantly reduces the com-
munication cost without hurting convergence time
for both iid and non-iid data.

## Code Organization

- **main.py** generates the main result of the simulation and in the following part you can comment some baseline algorithms to prevent simulating them. For instance this part adds Async-Gossip simulation to the final result.
```python
    ########async_goss###############
    with mp.Pool(repeat_simulation) as pool:
        res = pool.starmap_async(my_func, [(async_goss,
                                           (mnist, feature, label, exp, cte, iteration)) for run in range(repeat_simulation)]).get()
    result.append(res)
    print("All done -", "async_goss- ", repeat_simulation, "times")
    ########async_goss###############
```
- The configuration of the simulation can be set at the beginning of the **main.py** as shown in the following:
```python
###############config_variables###############
config_file_name = '100'
iteration = 10 ** 6
repeat_simulation = 20
iid = False
mnist = True
delay_shift = 1  # delays would be shifted exponential distributed RV with shift = delay_shift
av_period = 5  # P for PGA algorithms
multi_gossip = 5  # number of consecutive gossips in multi gossip PGA
sync_limit = 800 #single Digest_H
sync_limit_tree = 800 #multi_Digest_H
Gossip_H = 800 #Gossip_H
simulation_result_file_name = "simulation_result"
###############config_variables###############
```

- **config_file_name** identifies the network graph configuration that is defined in the config folder such that the first line is a dictionary with key=node and value=list of neighbors,
second line is a dictionary with key=node and value=list of average delay to neighbors,
Third line is a list of the lag of each node in computing SGD.
- Once you run **main.py** the simulation data will be saved at **result/simulation_result_file_name**.
- For the purpose of visualization use **visualization.py** where you first set the config as follows.
```python
###############config_variables###############
data_file_name = 'simulation_result'
final_figure_name = 'l_t_multist'
mnist = True
###############config_variables###############
```
You can make for different figure for each data file:
1. loss ove iterations
2. loss over wall-clock time
3. communication over iterations
4. communication over wall-clock time

In order to make any od the four above, you need to uncomment the related 
part at the end of the **visualization.py** file. For instance, if you uncomment
the following part, tou will have "loss over wall-clock time" figure.

```python
###############loss_time###############
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
if mnist:
    plt.ylim(0.2,0.6)
else:
    plt.ylim(0.13, 0.2)
ax.set_xlabel('Time/s')
ax.set_ylabel('Global loss')
plt.xlim(0,1000)
###############loss_time###############
```
Finally the figure will be saved at **figures/final_figure_name.pdf**.
