import json
import numpy as np

###############config_variables###############
num_of_nodes = 10
prob = .3
delay_range=[1, 10]
###############config_variables###############

gap = [0 for i in range(num_of_nodes)]
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

node_connection = {}
connection_delay = {}
for node in range(num_of_nodes):
    node_connection[node] = []
    connection_delay[node] = []

for node in range(num_of_nodes):
    for n in range(node + 1, num_of_nodes):
        if np.random.choice([0, 1], p=[1-prob, prob]) == 1:
            delay = np.random.choice(range(delay_range[0], delay_range[1]))
            node_connection[node].append(n)
            connection_delay[node].append(delay)
            node_connection[n].append(node)
            connection_delay[n].append(delay)
print(node_connection)
print(connection_delay)
#%%
with open('config/%s' %num_of_nodes, 'w') as f:
    f.write(json.dumps(node_connection))
    f.write("\n")
    f.write(json.dumps(connection_delay,cls=NpEncoder))
    f.write("\n")
    f.write(json.dumps(gap))