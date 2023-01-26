import numpy as np


class Network:
    def __init__(self, d, all_node, edge, action, pre_action):
        self.all_node = all_node
        self.edge = edge
        self.action = action
        self.pre_action = pre_action
        self.visited = []
        self.x_agg = np.zeros(d)
        self.q = np.ones(len(self.all_node))
        self.p = np.ones(len(self.all_node)) / len(self.all_node)
        self.m = 10

    def async_gossip(self, t):
        c = 0
        for node in range(len(self.all_node)):
            node_obj = self.all_node[node]
            if len(node_obj.async_agg) == 0:
                neighb = np.random.choice(self.edge[node])
                neighb_obj = self.all_node[neighb]
                if len(neighb_obj.async_agg) == 0:
                    c += 2
                    future_x = (node_obj.x + neighb_obj.x) / 2
                    node_obj.async_agg.append(
                        [t - (-neighb_obj.observe_delay(neighb_obj.neighbor.index(node)) // 1), future_x,
                         np.copy(node_obj.x)])
                    neighb_obj.async_agg.append(
                        [t - (-node_obj.observe_delay(node_obj.neighbor.index(neighb)) // 1), future_x,
                         np.copy(neighb_obj.x)])
        for node in range(len(self.all_node)):
            node_obj = self.all_node[node]
            node_obj.async_agg.sort(key=lambda tup: tup[0])
            counter = 0
            for update in node_obj.async_agg:
                #             print(update[0])
                if update[0] == t:
                    node_obj.x = update[1] + (node_obj.x - update[2])
                    counter += 1
                elif update[0] < t:
                    print("***************ERROR*****************")
                else:
                    break
            del node_obj.async_agg[:counter]
        return np.average([node.x for node in self.all_node], axis=0), c

    def gossip(self, constraint):
        x_values = []
        for node in self.all_node:
            x_values.append(np.copy(node.x))
        if constraint:
            for node in [np.random.choice(len(self.all_node))]:
                neighb = np.random.choice(self.edge[node])
                neighb_obj = self.all_node[neighb]
                node_obj = self.all_node[node]
                node_obj.x = (node_obj.x + x_values[neighb]) / 2
                time = neighb_obj.observe_delay(neighb_obj.neighbor.index(node))
                c = 1
        else:
            x_values = np.array(x_values)
            w = []
            for node in range(len(self.all_node)):
                not_norm_w = self.gossip_weight(node)
                w.append(not_norm_w / not_norm_w.sum())
            w = np.array(w)
            res = []
            for i in range(len(x_values[0])):
                res.append(np.matmul(w, x_values[:, i, :]))
            final = []
            for i in range(len(self.all_node)):
                final.append([res[j][i] for j in range(len(x_values[0]))])
            final = np.array(final)
            times = []
            for node in range(len(self.all_node)):
                node_obj = self.all_node[node]
                node_obj.x = np.copy(final[node])
                times.append(max([node_obj.observe_delay(i) for i in range(len(self.edge[node]))]))
            time = max(times)
            edge_size = 0
            for node in self.edge:
                edge_size += len(self.edge[node])
            c = edge_size
        return np.average([node.x for node in self.all_node], axis=0), time, c

    def gossip_weight(self, node):
        return np.array([int(i in self.edge[node]) for i in range(len(self.all_node))]) + np.array(
            [int(i == node) for i in range(len(self.all_node))])

    def final(self):
        return np.average([node.x_hat for node in self.all_node], axis=0,
                          weights=[len(node.feature) for node in self.all_node])

    def rw(self, node):
        random_select = np.random.choice(self.edge[node])
        p = np.random.uniform()
        if p <= min(1, len(self.edge[node]) / len(self.edge[random_select])):
            return random_select
        return node
