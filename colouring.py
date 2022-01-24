
import networkx as nx
import numpy as np
import tensorflow as tf
from ordered_set import OrderedSet


# feature_vector_shape = [1]
# k = 2


def randomNPGraph(n, p, diagonal=True, undirected=True):
    adjM = np.random.binomial(1, p, (n, n))
    if diagonal:
        for i in range(len(adjM)):
            adjM[i, i] = 1
    if undirected:
        xy = np.mgrid[0:n:1, 0:n:1].reshape(2, -1).T.reshape(n, n, 2)
        adjM = np.where(xy[..., 1] > xy[..., 0], adjM, adjM.T)
    return adjM


def randomGraphColoring(n, m, max_color=None):
    if max_color is None:
        max_color = m
    coloring = np.zeros((n, max_color))
    indices = list((np.arange(n), np.random.randint(m, size=n)))
    coloring[tuple(indices)] = 1
    return coloring


def checkGraphColoringError(adjM, coloring):
    neighbours = [np.where(adjM[i] == 1) for i in range(len(adjM))]
    errors = np.array(
        [[np.sum(coloring[i] * coloring[j]) if i != j and j in neighbours[i][0] else 0 for j in range(len(adjM))] for i
         in range(len(adjM))])
    sum_of_errors = np.sum(errors) / 2
    return sum_of_errors


def checkIfGraphConnected(adjM):
    G = nx.from_numpy_matrix(adjM)
    return nx.is_connected(G)


def generateGraphColoring(size, n_range, m_range, p_range):
    m_max = m_range[1 ] -1
    graphs = []
    while True:
        n = np.random.randint(n_range[0], n_range[1])
        m = np.random.randint(m_range[0], m_range[1])
        p = np.random.uniform(p_range[0], p_range[1])
        # print("n: " + str(n) +", m: " +str(m)+ ", p: " + str(p))
        NPGraph = randomNPGraph(n, p)
        connected = checkIfGraphConnected(NPGraph)
        if not connected:
            continue
        coloring = randomGraphColoring(n, m, max_color=m_max)
        coloringError = checkGraphColoringError(NPGraph, coloring)
        coloringError = 0.0 if coloringError ==0 else 1.0
        parts = [OrderedSet([i]) for i in range(len(NPGraph))]
        graph = [NPGraph, coloring, coloringError, parts]
        graphs.append(graph)
        if len(graphs) >= size:
            break
    return graphs


# n = 7  # nodes
# m = 4  # colors
# p = 0.4  # edge probability
# NPGraph = randomNPGraph(n, p)
# coloring = randomGraphColoring(n, m, max_color=None)
#
# connected = checkIfGraphConnected(NPGraph)
# coloringError = checkGraphColoringError(NPGraph, coloring)
# print(coloring)
# print(connected)
# print(coloringError)
def createTensors(graphs):
    Y=[tf.one_hot(int(y),2) for y in graphs[2]]
    print(Y)
    adj = [tf.Variable(adj) for adj in graphs[0]]
    X = [tf.cast(tf.Variable(y), tf.float32) for y in graphs[1]]
    return X, adj, Y

def get_graph_coloring_data():
    channels_in = 5
    data_size = 600
    graphs = list(zip(*generateGraphColoring(data_size, (3, 7), (channels_in, channels_in +1), (0.2, 0.5))))
    graphsValid = list(zip(*generateGraphColoring(250, (3, 7), (channels_in, channels_in +1), (0.2, 0.5))))

    # Xval, Yval = model.createTensors(graphsValid[1], graphsValid[2])
    # model.add_valid(Xval, Yval, graphsValid[0], graphsValid[3])

    # print(graphs)
    uq =np.unique(graphs[2], return_counts=True)
    print(np.unique(graphs[2], return_counts=True))
    classW =  data_size /(2 * uq[1])

    classWdict = {clazz :weight for clazz, weight in zip(uq[0], classW)}
    # model.class_weights = classWdict
    print(classWdict)
    # adjM = graphs[0]
    # X = graphs[1]
    # Y = graphs[2]
    # parts = graphs[3]
    train_tensors, test_tensors = createTensors(graphs), createTensors(graphsValid)
    a=0
    return train_tensors, test_tensors


    # X ,Y = model.createTensors(X ,Y)
    # model.fit(X, Y, adjM, parts, 1000)