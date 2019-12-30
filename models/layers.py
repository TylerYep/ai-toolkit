import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
    We multiply this against a weight matrix of shape (D, M) where
    D = \prod_i d_i

    Inputs:
    x - Input data, of shape (N, d_1, ..., d_k)
    w - Weights, of shape (D, M)
    b - Biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = x.reshape(x.shape[0], -1).dot(w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.reshape(x.shape[0], -1).T.dot(dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    x = cache
    dx = np.where(x > 0, dout, 0)
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.
    Uses minibatch statistics to compute the mean and variance, use
    these statistics to normalize the incoming data, and scale and
    shift the normalized data using gamma and beta.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the mean
    and variance of each feature, and these averages are used to normalize data
    at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that though you should be keeping track of the running
    variance, you should normalize the data based on the standard
    deviation (square root of variance) instead!
    Referencing the original paper (https://arxiv.org/abs/1502.03167)
    might prove to be helpful.

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7 implementation
    of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        # Compute output
        mu = x.mean(axis=0)
        xc = x - mu
        var = np.mean(xc ** 2, axis=0)
        std = np.sqrt(var + eps)
        xn = xc / std
        out = gamma * xn + beta

        cache = (mode, x, gamma, xc, std, xn, out)

        # Update running average of mean
        running_mean *= momentum
        running_mean += (1 - momentum) * mu

        # Update running average of variance
        running_var *= momentum
        running_var += (1 - momentum) * var

    elif mode == 'test':
        # Using running mean and variance to normalize
        std = np.sqrt(running_var + eps)
        xn = (x - running_mean) / std
        out = gamma * xn + beta
        cache = (mode, x, xn, gamma, beta, std)

    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    mode = cache[0]
    if mode == 'train':
        mode, x, gamma, xc, std, xn, out = cache

        N = x.shape[0]
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(xn * dout, axis=0)
        dxn = gamma * dout
        dxc = dxn / std
        dstd = -np.sum((dxn * xc) / (std * std), axis=0)
        dvar = 0.5 * dstd / std
        dxc += (2.0 / N) * xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / N

    elif mode == 'test':
        mode, x, xn, gamma, beta, std = cache
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(xn * dout, axis=0)
        dxn = gamma * dout
        dx = dxn / std

    else:
        raise ValueError(mode)

    return dx, dgamma, dbeta


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    N, C, H, W = x.shape
    x_flat = x.transpose(0, 2, 3, 1).reshape(-1, C)
    out_flat, cache = batchnorm_forward(x_flat, gamma, beta, bn_param)
    out = out_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    N, C, H, W = dout.shape
    dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, C)
    dx_flat, dgamma, dbeta = batchnorm_backward(dout_flat, cache)
    dx = dx_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Uses transpose matrix transformations to reuse batch norm code.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    eps = ln_param.get('eps', 1e-5)

    cache = {}
    x = x.T
    sample_mean = np.mean(x, axis=0)
    sample_var = np.var(x, axis=0)
    x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)
    x_hat = x_hat.T
    out = gamma * x_hat + beta

    cache['x_hat'] = x_hat
    cache['gamma'] = gamma
    cache['v'] = np.sqrt(sample_var + eps)

    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    x_hat = cache['x_hat']
    gamma = cache['gamma']
    v = cache['v']

    dx_hat = dout * gamma
    dgamma = np.sum(dout * x_hat, axis=0)
    dbeta = np.sum(dout, axis=0)

    x_hat = x_hat.T
    dx_hat = dx_hat.T
    dx = (dx_hat - np.mean(dx_hat, axis=0) - x_hat * np.mean(dx_hat * x_hat, axis=0)) / v
    dx = dx.T

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask, out = None, None
    if mode == 'train':
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask

    elif mode == 'test':
        out = x

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        dx = dout * mask

    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.


    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modify the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    pad = conv_param['pad']
    stride = conv_param['stride']
    H_prime = 1 + (H + 2 * pad - HH) // stride
    W_prime = 1 + (W + 2 * pad - WW) // stride
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    out = np.zeros((N, F, H_prime, W_prime))

    for n in range(N):
        for f in range(F):
            for i in range(0, H_prime*stride, stride):
                for j in range(0, W_prime*stride, stride):
                    out[n, f, i//stride, j//stride] = \
                        np.sum(w[f] * x_padded[n, :, i:i+HH, j:j+WW]) + b[f]
    cache = (x, w, b, conv_param)

    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    x, w, b, conv_param = cache
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    pad = conv_param['pad']
    stride = conv_param['stride']
    _, _, H_prime, W_prime = dout.shape
    padding = ((0, 0), (0, 0), (pad, pad), (pad, pad))
    x_padded = np.pad(x, padding, mode='constant')
    dx_padded = np.pad(dx, padding, mode='constant')

    for n in range(N):
        for f in range(F):
            for i in range(0, H_prime*stride, stride):
                for j in range(0, W_prime*stride, stride):
                    do = dout[n, f, i//stride, j//stride]
                    dx_padded[n, :, i:i+HH, j:j+WW] += do * w[f]
                    dw[f] += do * x_padded[n, :, i:i+HH, j:j+WW]
                    db[f] += do
    dx = dx_padded[:, :, pad:-pad, pad:-pad]

    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H, W = x.shape
    H_prime = 1 + (H - pool_height) // stride
    W_prime = 1 + (W - pool_width) // stride

    out = np.zeros((N, C, H_prime, W_prime))
    for n in range(N):
        for c in range(C):
            for i in range(0, H_prime*stride, stride):
                for j in range(0, W_prime*stride, stride):
                    out[n, c, i//stride, j//stride] = \
                        np.amax(x[n, c, i:i+pool_height, j:j+pool_width])
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    x, pool_param = cache
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H, W = x.shape
    H_prime = 1 + (H - pool_height) // stride
    W_prime = 1 + (W - pool_width) // stride

    dx = np.zeros(x.shape)
    for n in range(N):
        for c in range(C):
            for i in range(0, H_prime*stride, stride):
                for j in range(0, W_prime*stride, stride):
                    a = x[n, c, i:i+pool_height, j:j+pool_width]
                    k = np.unravel_index(a.argmax(), a.shape)
                    dx[n, c, i:i+pool_height, j:j+pool_width][k] = dout[n, c, i//stride, j//stride]
    return dx


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner
    identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    eps = gn_param.get('eps', 1e-5)
    N, C, H, W = x.shape
    gamma = gamma.reshape((1, C, 1, 1))
    beta = beta.reshape((1, C, 1, 1))

    x = x.reshape(N * G, -1).T
    sample_mean = np.mean(x, axis=0)
    sample_var = np.var(x, axis=0)
    x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)
    x_hat = x_hat.T.reshape(N, C, H, W)
    out = gamma * x_hat + beta

    cache = {}
    cache['x_hat'] = x_hat
    cache['gamma'] = gamma
    cache['v'] = np.sqrt(sample_var + eps)
    cache['G'] = G

    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    x_hat = cache['x_hat']
    gamma = cache['gamma']
    G = cache['G']
    v = cache['v']
    N, C, H, W = dout.shape

    dx_hat = dout * gamma
    dgamma = np.sum(dout * x_hat, axis=(0, 2, 3), keepdims=True)
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)

    x_hat = x_hat.reshape(N * G, -1).T
    dx_hat = dx_hat.reshape(N * G, -1).T
    dx = (dx_hat - np.mean(dx_hat, axis=0) - x_hat * np.mean(dx_hat * x_hat, axis=0)) / v
    dx = dx.T.reshape(N, C, H, W)

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
