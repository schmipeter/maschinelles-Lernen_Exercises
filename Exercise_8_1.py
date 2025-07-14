def conv_out_dim(n, k, p=0, s=1):
    """Return output length of a 1-D discrete convolution."""
    return (n + 2 * p - k) // s + 1


# 2-D wrapper
def conv_out_shape(input_hw, kernel_hw, padding_hw=(0, 0), stride_hw=(1, 1)):
    h = conv_out_dim(input_hw[0], kernel_hw[0], padding_hw[0], stride_hw[0])
    w = conv_out_dim(input_hw[1], kernel_hw[1], padding_hw[1], stride_hw[1])
    return h, w


# Example from Figure 8.1: 5×5 input, 2×3 kernel, no padding, stride 1
print(conv_out_shape((5, 5), (2, 3)))  # (4, 3)

# Same-size convolution with 3×3 kernel
print(conv_out_shape((5, 5), (3, 3), padding_hw=(1, 1)))  # (5, 5)
