def call(self, act, pose):



def EM_routing(self, act_in, X):
    if self.transPOSE:
        X = X.transpose([0, 1, 3, 2])
    c1, c2 = X.shape[2], X.shape[3]
    c3 = self.transitions.shape[3]
    # this estimate of R is an initial E-step
    R = tf.Variable(np.ones((1, self.n_capsules_in, self.n_capsules_out)) / self.n_capsules_out)
    V = self.voting(X)
    act_out, mu, sigma = self.M(act_in, R, V)
    i = tf.constant(1)
    cond = lambda n: tf.less(n, 3)
    r = tf.while_loop(cond, body, [i, R, act_out, mu, sigma])
    R = self.E(act_out, V, mu, sigma)
    act_out, mu, sigma = self.M(act_in, R, V)


return act_out, mu.reshape(X.shape[0], self.n_capsules_out, c1, c3)


def voting(self, X):
    return tf.matmul(tf.cast(X[:, :, tf.newaxis, :, :], tf.float32), self.transitions)


def E(self, act_out, V, mu, sigma):
    distribution = tfd.Normal(mu, sigma)
    prob = distribution.prob(V)
    raw_R = act_out[:, tf.newaxis, :, tf.newaxis, tf.newaxis] * prob
    return tf.reduce_sum(tf.math.divide_no_nan(raw_R, tf.linalg.norm(raw_R, ord=1, keepdims=True)), axis=[-2, -1])


def M(self, act_in, R, V):
    odds = act_in[:, :, tf.newaxis] * R
    r = tf.math.divide_no_nan(odds, np.sum(odds, axis=1, keepdims=True))[:, :, :, np.newaxis, np.newaxis]
    mu = tf.math.reduce_sum(r * V, axis=1, keepdims=True)
    sigma = tf.math.sqrt(tf.reduce_sum(r * (V - mu) ** 2, axis=1, keepdims=True))
    cost1 = r * tf.math.log(sigma)
    cost2 = tf.constant(1 / 2 + tf.math.log(2 * math.pi) / 2)
    cost = tf.math.reduce_sum(cost1 + cost2, axis=1)
    '''cost = tf.math.reduce_sum(
        r * tf.math.log(sigma) + tf.constant(1 / 2 + tf.math.log(2 * math.pi) / 2),
        axis=1)'''
    ac1 = self.lamb
    ac2 = tf.math.reduce_sum(r, axis=[1, -2, -1])
    ac3 = tf.math.reduce_sum(cost, axis=[2, 3])
    ac4 = self.bias_a - self.bias_b * ac2 - ac3
    act_out = tf.nn.softmax(ac1 * ac4)
    # act_out = tf.math.sigmoid(ac1 * ac4)
    '''
    act_out = tf.math.sigmoid(self.lamb * (self.bias_a - self.bias_b *
                                      tf.math.reduce_sum(r, axis=1) -
                                      tf.math.reduce_sum(cost, axis=[2, 3])))
    '''
    return act_out, mu, sigma

