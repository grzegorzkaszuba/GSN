import numpy as np
import networkx as nx
import tensorflow as tf
import pickle


def scipy_to_tf_sparse(sp):
    coo = sp.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


G = nx.Graph()
# load data
data_adj = np.loadtxt('ENZYMES_A.txt', delimiter=',').astype(int)
data_node_att = np.loadtxt('ENZYMES_node_attributes.txt', delimiter=',')
data_node_label = np.loadtxt('ENZYMES_node_labels.txt', delimiter=',').astype(int)
data_graph_indicator = np.loadtxt('ENZYMES_graph_indicator.txt', delimiter=',').astype(int)
data_graph_labels = np.loadtxt('ENZYMES_graph_labels.txt', delimiter=',').astype(int)


data_tuple = list(map(tuple, data_adj))
print(len(data_tuple))
print(data_tuple[0])

# add edges
G.add_edges_from(data_tuple)
# add node attributes
for i in range(data_node_att.shape[0]):
    G.add_node(i+1, feature = data_node_att[i])
    G.add_node(i+1, label = data_node_label[i])
#edgelist = list(G.network.edges(data=True))

print(G.number_of_nodes())
print(G.number_of_edges())

# split into graphs
graph_num = 600
node_list = np.arange(data_graph_indicator.shape[0])+1
graphs = []
node_num_list = []
for i in range(graph_num):
    # find the nodes for each graph
    nodes = node_list[data_graph_indicator==i+1]
    G_sub = G.subgraph(nodes)
    graphs.append(G_sub)
    G_sub.graph['label'] = data_graph_labels[i]
    # print('nodes', G_sub.number_of_nodes())
    # print('edges', G_sub.number_of_edges())
    # print('label', G_sub.graph)
    node_num_list.append(G_sub.number_of_nodes())
print('sum', sum(node_num_list))
print('average', sum(node_num_list)/len(node_num_list))
print('min', min(node_num_list))
print('all', len(node_num_list))
node_num_list = np.array(node_num_list)
print('selected', len(node_num_list[node_num_list>10]))
# print(graphs[0].nodes(data=True)[0][1]['feature'])
# print(graphs[0].nodes())
keys = tuple(graphs[0].nodes())
# print(nx.get_node_attributes(graphs[0], 'feature'))
dictionary = nx.get_node_attributes(graphs[0], 'feature')
print('keys', keys)
# print('keys from dict', list(dictionary.keys()))
# print('valuse from dict', list(dictionary.values()))
print('max nodes: ', max(node_num_list))

features = np.zeros((len(dictionary), list(dictionary.values())[0].shape[0]))
for i in range(len(dictionary)):
    features[i,:] = list(dictionary.values())[i]
#print(features)
#print(features.shape)

#data = []
attribute_list = []
adjacency_list = []

def stack_padding(it):

    def resize(row, size):
        new = np.array(row)
        new.resize(size)
        return new

    # find longest row length
    row_length = max(it, key=len).__len__()
    mat = np.array( [resize(row, row_length) for row in it] )

    return mat

check = True
for g in graphs:
    attributes = np.array(data_node_att[[n-1 for n in tuple(g.nodes)], :], dtype='float32')
    if max(attributes.shape) > 40:
        continue
    #attributes2 = np.zeros((40, 18))
    #attributes2[:attributes.shape[0], :attributes.shape[1]] += attributes
    attributes = tf.constant(attributes, dtype='float32')
    #adjacency_matrix = tf.constant(nx.linalg.graphmatrix.adjacency_matrix(g).toarray(), dtype='float32')
    adjacency_matrix = nx.linalg.graphmatrix.adjacency_matrix(g).toarray()
    #adjacency_matrix2 = np.zeros((40, 40))
    #adjacency_matrix2[:adjacency_matrix.shape[0], :adjacency_matrix.shape[1]] += adjacency_matrix
    adjacency_matrix = tf.constant(adjacency_matrix)
    #adjacency_matrix = scipy_to_tf_sparse(nx.linalg.graphmatrix.adjacency_matrix(g))
    #data.append([attributes, adjacency_matrix])
    attribute_list.append(attributes)
    adjacency_list.append(adjacency_matrix)

print(len(attribute_list))


with open('ENZYMES_graph_labels.txt', 'r') as f:
    labels = [int(line.strip())-1 for line in f.readlines()]

lab = tf.one_hot(labels, 6)

with open('labels.pickle', 'wb') as f:
    pickle.dump(lab, f)

with open('attributes.pickle', 'wb') as f:
    pickle.dump(attribute_list, f)

with open('adjacency.pickle', 'wb') as f:
    pickle.dump(adjacency_list, f)