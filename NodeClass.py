from Functions import *

class Node:
    def __init__(self, d, d_hat, feature, label, neighbor, lag, delay_dist, stream, delay_shift):
        self.delay_shift = delay_shift
        self.neighbor = neighbor
        self.feature = feature
        self.label = label
        self.lag = lag
        self.x = np.zeros(d)
        self.x_tau = [np.zeros(d) for st in range(stream)]
        self.x_pre = np.zeros(d)
        self.t_pre = 0
        self.x_hat = np.zeros(d_hat)
        self.tau = [0 for st in range(stream)]
        self.count = [0 for i in range(len(neighbor))]
        self.utiliy = [0 for i in range(len(neighbor))]
        self.delay_dist = delay_dist
        self.traverse_parent = [None for st in range(stream)]
        self.async_agg = []

    def observe_delay(self, a_index):
        return self.delay_shift + np.random.exponential(self.delay_dist[a_index])

    def traverse_select(self, from_node, all_node, visited, st):
        if self.traverse_parent[st] is None:
            self.traverse_parent[st] = from_node
        pv = self.traverse_parent[st]
        Nv = self.neighbor
        for node in set(Nv).intersection(all_node):
            if node not in visited:
                a_index = self.neighbor.index(node)
                del_observed = self.observe_delay(a_index)
                return self.neighbor[a_index], del_observed
        for node in all_node:
            if node not in visited:
                a_index = self.neighbor.index(pv)
                del_observed = self.observe_delay(a_index)
                self.traverse_parent[st] = None
                return self.neighbor[a_index], del_observed
        self.traverse_parent[st] = None
        return "Done", 0

    def local_sgd(self,mnist,total_data,t, learning_rate):
        i_feature, i_label = uniform_data_catch(self.feature, self.label)
        temp = np.copy(self.x)
        self.x, gr = logistic_regression(mnist,total_data,self.x, i_feature, i_label, learning_rate, num_steps=1)
        self.x_hat = update_x_hat(self.x, self.x_hat, t)
        return gr

    def synchronize(self, t, x_agg, total_data, st):
        data_size = len(self.feature)
        x_agg = x_agg + data_size / total_data * (self.x - self.x_pre) + self.x_pre - self.x_tau[st]
        self.t_pre = t + 1
        self.x_tau[st] = np.copy(x_agg)
        self.x = np.copy(x_agg)
        self.x_pre = np.copy(x_agg)
        self.tau[st] = t + 1
        return x_agg


