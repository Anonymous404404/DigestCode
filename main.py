import pandas as pd
import multiprocessing as mp
from tensorflow.keras.datasets import mnist as mnist_dataset
import json
from NetworkClass import *
from NodeClass import *
from DSGDAlg import *
import os

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



cte = False
exp = 1
# load topology from config folder
with open("config/%s" % config_file_name, "r") as f:
    fp = f.readlines()
    node_connection = {int(k): v for k, v in json.loads(fp[0]).items()}
    connection_delay = {int(k): v for k, v in json.loads(fp[1]).items()}
    gap = json.loads(fp[2])
num_of_node = len(node_connection)
# data:
if mnist:
    (x_train, y_train), (x_test, y_test) = mnist_dataset.load_data()
    x_train = x_train / 255.0
    n_train, num_row, num_col = x_train.shape
    feature_pre_shuffle = np.reshape(x_train, (n_train, num_row * num_col))
    label_pre_shuffle = pd.get_dummies(y_train).values
    d = (label_pre_shuffle.shape[1], feature_pre_shuffle.shape[1])
    d_hat = (4, label_pre_shuffle.shape[1], feature_pre_shuffle.shape[1])
else:
    d = 300
    d_hat = (4, d)
    url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a'
    df_train = pd.read_csv(url, names=['train'])
    url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a.t'
    df_test = pd.read_csv(url, names=['test'])
    feature_pre_shuffle, label_pre_shuffle = fetch(df_train, 'train', d)

shuffler = np.random.permutation(len(feature_pre_shuffle))
feature = feature_pre_shuffle[shuffler]
label = label_pre_shuffle[shuffler]
total_data = len(label)
data_split = []
guid_to_split = range(0, total_data + 1, total_data // num_of_node)
for node in range(num_of_node):
    data_split.append((guid_to_split[node], guid_to_split[node + 1]))

if not iid:
    if mnist:
        boolArr = np.argmax(label, axis=1) == 0
        x = feature[boolArr]
        y = label[boolArr]
        data_split[0] = (0, len(y))
        for n in range(1, 10):
            boolArr = np.argmax(label, axis=1) == n
            x1 = feature[boolArr]
            y1 = label[boolArr]
            x = np.vstack((x, x1))
            if mnist:
                y = np.vstack((y, y1))
            else:
                y = np.hstack((y, y1))
        feature = x
        label = y
        if num_of_node == 100:
            data_split = [(0, 10)]
            num_of_chunk = 10
            q = 2.04
            for n in range(1, num_of_chunk):
                data_split.append((data_split[n - 1][1], int(data_split[0][1] * q ** n)))
            data_split[-1] = (data_split[-1][0], total_data // num_of_chunk)
            for c in range(1, num_of_chunk):
                for n in range(num_of_chunk):
                    data_split.append((data_split[c * num_of_chunk + n - 1][1],
                                       data_split[c * num_of_chunk + n - 1][1] + data_split[n % num_of_chunk][1] -
                                       data_split[n % num_of_chunk][0]))
        else:
            data_split = [(0, 500)]
            q = 1.7
            for n in range(1, num_of_node):
                data_split.append((data_split[n - 1][1], int(data_split[0][1] * q ** n)))
            data_split[-1] = (data_split[-1][0], total_data)


    else:
        if num_of_node == 100:
            data_split = [(0, 10)]
            num_of_chunk = 10
            q = 1.9936
            for n in range(1, num_of_chunk):
                data_split.append((data_split[n - 1][1], int(data_split[0][1] * q ** n)))
            data_split[-1] = (data_split[-1][0], total_data // num_of_chunk)
            for c in range(1, num_of_chunk):
                for n in range(num_of_chunk):
                    data_split.append((data_split[c * num_of_chunk + n - 1][1],
                                       data_split[c * num_of_chunk + n - 1][1] + data_split[n % num_of_chunk][1] -
                                       data_split[n % num_of_chunk][0]))
        else:
            data_split = [(0, 500)]
            q = 1.67
            for n in range(1, num_of_node):
                data_split.append((data_split[n - 1][1], int(data_split[0][1] * q ** n)))
            data_split[-1] = (data_split[-1][0], total_data)


def my_func(f, args):
    network = Network(d, [], node_connection, 0, 0)
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    for node in range(num_of_node):
        network.all_node.append(Node(d, d_hat, feature[data_split[node][0]:data_split[node][1]],
                                     label[data_split[node][0]:data_split[node][1]],
                                     node_connection[node],
                                     gap[node], connection_delay[node], stream, delay_shift))
    return f(*args, network)


if __name__ == '__main__':
    result = []

    ########single-stream###############
    stream = 1
    with mp.Pool(repeat_simulation) as pool:
        res = pool.starmap_async(my_func, [(Digest,
                                           (mnist, feature, label, d, exp, cte, sync_limit, stream, iteration)) for run in range(repeat_simulation)]).get()
    result.append(res)
    print("All done -", "Digest_single-stream- ", repeat_simulation, "times")
    ########single-stream###############

    ########multi-stream###############
    a = floyd_warshall(node_connection, connection_delay)
    print("max_path_lenght=", a[2], "center=", a[1])
    streams_nodes = find_streams(a[0], a[1], -1)[-1]
    stream = len(streams_nodes)
    for sync_limit_tree_instance in sync_limit_tree:
        with mp.Pool(repeat_simulation) as pool:
            res = pool.starmap_async(my_func, [(Digest_tree,
                                                   (mnist, feature, label, d, exp, cte, sync_limit_tree_instance,
                                                    streams_nodes, iteration)) for run in range(repeat_simulation)]).get()
        result.append(res)
        print("All done -", "Digest_multi-stream- ", repeat_simulation, "times")
    ########multi-stream###############

    ########async_goss###############
    stream = 1
    with mp.Pool(repeat_simulation) as pool:
        res = pool.starmap_async(my_func, [(async_goss,
                                           (mnist, feature, label, exp, cte, iteration)) for run in range(repeat_simulation)]).get()
    result.append(res)
    print("All done -", "async_goss- ", repeat_simulation, "times")
    ########async_goss###############

    ########Gossip###############
    stream = 1
    with mp.Pool(repeat_simulation) as pool:
        res = pool.starmap_async(my_func, [(goss,
                                           (mnist, feature, label, exp, cte, iteration, Gossip_H, False)) for run in range(repeat_simulation)]).get()
    result.append(res)
    print("All done -", "goss- ", repeat_simulation, "times")
    ########Gossip###############

    ########One-link-Gossip###############
    stream = 1
    with mp.Pool(repeat_simulation) as pool:
        res = pool.starmap_async(my_func, [(goss,
                                           (mnist, feature, label, exp, cte, iteration, Gossip_H, True)) for run in range(repeat_simulation)]).get()
    result.append(res)
    print("All done -", "One-link-Gossip- ", repeat_simulation, "times")
    ########One-link-Gossip###############

    ########Real-avg-PGA-Gossip###############
    stream = 1
    with mp.Pool(repeat_simulation) as pool:
        res = pool.starmap_async(my_func, [(goss_periodic_real_av,
                                           (mnist, feature, label, exp, cte, iteration, av_period)) for run in range(repeat_simulation)]).get()
    result.append(res)
    print("All done -", "Real-avg-PGA-Gossip- ", repeat_simulation, "times")
    ########Real-avg-PGA-Gossip###############

    ########multi-gossip-PGA-Gossip###############
    stream = 1
    with mp.Pool(repeat_simulation) as pool:
        res = pool.starmap_async(my_func, [(goss_periodic_av,
                                           (mnist, feature, label, exp, cte, iteration, av_period,multi_gossip)) for run in range(repeat_simulation)]).get()
    result.append(res)
    print("All done -", "multi-gossip-PGA-Gossip- ", repeat_simulation, "times")
    ########multi-gossip-PGA-Gossip###############



    with open("result/"+simulation_result_file_name, 'w') as f:
        f.write(json.dumps(result))