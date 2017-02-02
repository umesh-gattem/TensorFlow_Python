# TensorFlow_Normalization

## tf.nn.l2_normalize(x, dim, epsilon=1e-12, name=None)

Normalizes along dimension dim using an L2 norm.<br>
For a 1-D tensor with dim = 0, computes
```python
output = x / sqrt(max(sum(x**2), epsilon))
```
For x with more dimensions, independently normalizes each 1-D slice along dimension dim.<br>

### Args:
1. x: A Tensor.<br>
2. dim: Dimension along which to normalize. A scalar or a vector of integers.<br>
3. epsilon: A lower bound value for the norm. Will use sqrt(epsilon) as the divisor if norm < sqrt(epsilon).<br>
4. name: A name for this operation (optional).<br>

### Returns

A Tensor with the same shape as x.

## tf.nn.local_response_normalization(input, depth_radius=None, bias=None, alpha=None, beta=None, name=None)

Local Response Normalization.<br>
The 4-D input tensor is treated as a 3-D array of 1-D vectors (along the last dimension), 
and each vector is normalized independently. Within a given vector, 
each component is divided by the weighted, squared sum of inputs within depth_radius. In detail,

```python 
sqr_sum[a, b, c, d] = sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
output = input / (bias + alpha * sqr_sum) ** beta
```
### Args:
1. input: A Tensor. Must be one of the following types: float32, half. 4-D.<br>
2. depth_radius: An optional int. Defaults to 5. 0-D. Half-width of the 1-D normalization window.<br>
3. bias: An optional float. Defaults to 1. An offset (usually positive to avoid dividing by 0).<br>
4. alpha: An optional float. Defaults to 1. A scale factor, usually positive.<br>
5. beta: An optional float. Defaults to 0.5. An exponent.<br>
6. name: A name for the operation (optional).<br>

### Returns:

A Tensor. Has the same type as input.

## References

1. [What is Local Normalization Layer in CNN?](https://prateekvjoshi.com/2016/04/05/what-is-local-response-normalization-in-convolutional-neural-networks/)<br>
2. [Go through the section 3.3 in this link](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)<br>
3. [Tensorflow_Normalisation](https://www.tensorflow.org/versions/r1.0/api_docs/python/nn/normalization)<br>
