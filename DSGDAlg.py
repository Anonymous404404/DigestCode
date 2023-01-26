from Functions import *


# Single-stream
def Digest(mnist, feature, label, d, exp, cte, sync_limit, stream, iters, network):
    total_data = len(label)
    sync_mode = [False for i in range(stream)]
    num_of_node = len(network.all_node)
    action = list(range(0, num_of_node, num_of_node // stream))
    pre_action = list(range(0, num_of_node, num_of_node // stream))
    del_obs = [0 for i in range(stream)]
    al = [range(num_of_node) for i in range(stream)]
    net_x_agg = [np.zeros(d) for i in range(stream)]
    node = [None for i in range(stream)]
    vis = [[] for i in range(stream)]
    loss_over_t = []
    comm_over_t = []
    real_time_over_t = []
    real_time = 0
    comm = 0
    sync_moments = [[] for i in range(num_of_node)]
    for t in range(iters):
        if t % 1000== 0:
            final_x = network.final()
            real_time_over_t.append(real_time)
            comm_over_t.append(comm)
            loss_over_t.append([loss(mnist, feature, label, final_x[i]) for i in range(4)])
            print(loss_over_t[-1])
            print(real_time_over_t[-1])
            print(comm_over_t[-1])
        for nodee in network.all_node:
            end = t - nodee.lag + 1
            start = max(t - nodee.lag, 0)
            for sgd_round in range(start, end):
                nodee.local_sgd(mnist, total_data, sgd_round, learning_rate(sgd_round, exp, cte))
        real_time += 1
        for st in range(stream):
            if t % sync_limit == 0 or sync_mode[st]:
                if not sync_mode[st]:
                    sync_mode[st] = True
                    node[st] = network.all_node[action[st]]
                    net_x_agg[st] = node[st].synchronize(t, net_x_agg[st], total_data, st)
                    vis[st].append(action[st])
                    action[st], del_obs[st] = node[st].traverse_select(pre_action[st], al[st], vis[st], st)
                if del_obs[st] > 0:
                    del_obs[st] -= 1
                else:
                    sync_moments[action[st]].append(real_time)
                    comm += 1
                    vis[st].append(action[st])
                    node[st] = network.all_node[action[st]]
                    net_x_agg[st] = node[st].synchronize(t, net_x_agg[st], total_data, st)
                    action[st], del_obs[st] = node[st].traverse_select(pre_action[st], al[st], vis[st], st)
                    if action[st] == "Done":
                        sync_mode[st] = False
                        vis[st] = []
                        for i in al[st]:
                            network.all_node[i].traverse_parent[st] = None
                        action[st] = network.all_node.index(node[st])
            pre_action[st] = network.all_node.index(node[st])
    return [loss_over_t, real_time_over_t, comm_over_t,sync_moments]

#digest multi_stream
def Digest_tree(mnist, feature, label, d, exp, cte, sync_limit, stream_nodes, iters, network):
    stream = len(stream_nodes)
    print(stream_nodes)
    print(stream)
    total_data = len(label)
    sync_mode = [False for i in range(stream)]
    num_of_node = len(network.all_node)
    action = [np.random.choice(stream_nodes[st]) for st in range(stream)]
    pre_action = [np.random.choice(stream_nodes[st]) for st in range(stream)]
    del_obs = [0 for i in range(stream)]
    al = stream_nodes
    net_x_agg = [np.zeros(d) for i in range(stream)]
    node = [None for i in range(stream)]
    vis = [[] for i in range(stream)]
    loss_over_t = []
    comm_over_t = []
    real_time_over_t = []
    real_time = 0
    comm = 0
    sync_moments = [[] for i in range(num_of_node)]
    for t in range(iters):
        if t % 1000 == 0:
            final_x = network.final()
            real_time_over_t.append(real_time)
            comm_over_t.append(comm)
            loss_over_t.append([loss(mnist, feature, label, final_x[i]) for i in range(4)])
            print(loss_over_t[-1])
            print(real_time_over_t[-1])
            print(comm_over_t[-1])
        if True:
            for nodee in network.all_node:
                end = t - nodee.lag + 1
                start = max(t - nodee.lag, 0)
                for sgd_round in range(start, end):
                    nodee.local_sgd(mnist, total_data, sgd_round, learning_rate(sgd_round, exp, cte))
        real_time += 1
        for st in range(stream):
            if t % sync_limit == 0 or sync_mode[st]:
                if not sync_mode[st]:
                    sync_mode[st] = True
                    node[st] = network.all_node[action[st]]
                    net_x_agg[st] = node[st].synchronize(t, net_x_agg[st], total_data, st)
                    vis[st].append(action[st])
                    action[st], del_obs[st] = node[st].traverse_select(pre_action[st], al[st], vis[st], st)
                if del_obs[st] > 0:
                    del_obs[st] -= 1
                else:
                    sync_moments[action[st]].append(real_time)
                    comm += 1
                    vis[st].append(action[st])
                    node[st] = network.all_node[action[st]]
                    net_x_agg[st] = node[st].synchronize(t, net_x_agg[st], total_data, st)
                    action[st], del_obs[st] = node[st].traverse_select(pre_action[st], al[st], vis[st], st)
                    if action[st] == "Done":
                        sync_mode[st] = False
                        vis[st] = []
                        for i in al[st]:
                            network.all_node[i].traverse_parent[st] = None
                        action[st] = network.all_node.index(node[st])
            pre_action[st] = network.all_node.index(node[st])
    return [loss_over_t, real_time_over_t, comm_over_t,sync_moments]


def random_walk(mnist, feature, label, exp, cte, iters, network):
    total_data = len(label)
    loss_over_t = []
    real_time_over_t = []
    real_time = 0
    comm_over_t = []
    comm = 0
    for t in range(iters):
        if t % 1000 == 0:
            real_time_over_t.append(real_time)
            loss_over_t.append(loss(mnist, feature, label, network.x_agg))
            comm_over_t.append(comm)
            print(loss_over_t[-1])
            print(real_time_over_t[-1])
            print(comm_over_t[-1])
        node = network.all_node[network.action]
        node.x = np.copy(network.x_agg)
        node.local_sgd(mnist, total_data, t, learning_rate(t, exp, cte))
        network.x_agg = np.copy(node.x)
        network.action = network.rw(network.action)
        if network.all_node.index(node) != network.action:
            real_time += node.observe_delay(node.neighbor.index(network.action))
            comm += 1
        else:
            real_time += 1
    return [loss_over_t, real_time_over_t, comm_over_t]


def async_goss(mnist, feature, label, exp, cte, iters, network):
    total_data = len(label)
    loss_over_t = []
    real_time_over_t = []
    comm_over_t = []
    comm = 0
    real_time = 0
    for t in range(iters):
        if t % 1000 == 0:
            final_x = network.final()
            loss_over_t.append([loss(mnist, feature, label, final_x[i]) for i in range(4)])
            real_time_over_t.append(real_time)
            comm_over_t.append(comm)
            print(loss_over_t[-1])
            print(real_time_over_t[-1])
            print(comm_over_t[-1])
        for node in network.all_node:
            end = t - node.lag + 1
            start = max(t - node.lag, 0)
            for sgd_round in range(start, end):
                node.local_sgd(mnist, total_data, sgd_round, learning_rate(sgd_round, exp, cte))
        network.x_agg, c = network.async_gossip(t)
        comm += c
        real_time += 1
    return [loss_over_t, real_time_over_t, comm_over_t]


# Gossip
def goss(mnist, feature, label, exp, cte, iters, H, constraint, network):
    total_data = len(label)
    loss_over_t = []
    real_time_over_t = []
    comm_over_t = []
    comm = 0
    real_time = 0
    for t in range(iters):
        if t % 1000 == 0:
            final_x = network.final()
            loss_over_t.append([loss(mnist, feature, label, final_x[i]) for i in range(4)])
            real_time_over_t.append(real_time)
            comm_over_t.append(comm)
            print(loss_over_t[-1])
            print(real_time_over_t[-1])
            print(comm_over_t[-1])
        for node in network.all_node:
            end = t - node.lag + 1
            start = max(t - node.lag, 0)
            for sgd_round in range(start, end):
                node.local_sgd(mnist, total_data, sgd_round, learning_rate(sgd_round, exp, cte))
        if t % H == 0:
            network.x_agg, time, c = network.gossip(constraint)
            real_time += time
            comm += c
        else:
            real_time += 1
    return [loss_over_t, real_time_over_t, comm_over_t]


# Gossip- periodic averaging
def goss_periodic_real_av(mnist, feature, label, exp, cte, iters, av_period, network):
    total_data = len(label)
    loss_over_t = []
    real_time_over_t = []
    num_of_node = len(network.all_node)
    real_time = 0
    comm_over_t = []
    comm = 0
    for t in range(iters):
        if t % 1000 == 0:
            final_x = network.final()
            loss_over_t.append([loss(mnist, feature, label, final_x[i]) for i in range(4)])
            real_time_over_t.append(real_time)
            comm_over_t.append(comm)
            print(loss_over_t[-1])
            print(real_time_over_t[-1])
            print(comm_over_t[-1])
        for node in network.all_node:
            end = t - node.lag + 1
            start = max(t - node.lag, 0)
            for sgd_round in range(start, end):
                node.local_sgd(mnist, total_data, sgd_round, learning_rate(sgd_round, exp, cte))
        if t % av_period == 0:
            temp = np.average([node.x for node in network.all_node], axis=0,
                              weights=[len(node.feature) for node in network.all_node])
            for node in network.all_node:
                node.x = np.copy(temp)
            travers_time = 0
            c = 0
            while network.action != "Done":
                network.visited.append(network.action)
                c += 1
                node = network.all_node[network.action]
                network.action, del_observed = node.traverse_select(network.pre_action, range(num_of_node),
                                                                    network.visited,0)
                network.pre_action = network.all_node.index(node)
                travers_time += del_observed
            network.action = 0
            comm += c * 2
            real_time += 2 * travers_time
            network.visited = []
            for item in network.all_node:
                item.traverse_parent[0] = None
        else:
            network.x_agg, time, c = network.gossip(False)
            real_time += time
            comm += c
    return [loss_over_t, real_time_over_t, comm_over_t]


# Gossip- periodic multi gossip
def goss_periodic_av(mnist, feature, label, exp, cte, iters, av_period,multi_gossip, network):
    total_data = len(label)
    loss_over_t = []
    real_time_over_t = []
    num_of_node = len(network.all_node)
    real_time = 0
    comm_over_t = []
    comm = 0
    for t in range(iters):
        if t % 1000 == 0:
            final_x = network.final()
            loss_over_t.append([loss(mnist, feature, label, final_x[i]) for i in range(4)])
            real_time_over_t.append(real_time)
            comm_over_t.append(comm)
            print(loss_over_t[-1])
            print(real_time_over_t[-1])
            print(comm_over_t[-1])
        for node in network.all_node:
            end = t - node.lag + 1
            start = max(t - node.lag, 0)
            for sgd_round in range(start, end):
                node.local_sgd(mnist, total_data, sgd_round, learning_rate(sgd_round, exp, cte))
        if t % av_period == 0:
            for step in range(multi_gossip):
                network.x_agg, time, c = network.gossip(False)
                real_time += time
                comm += c
        else:
            network.x_agg, time, c = network.gossip(False)
            real_time += time
            comm += c
    return [loss_over_t, real_time_over_t, comm_over_t]