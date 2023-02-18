def transformer_count_params(d_model=512, d_ff=2048, layers=12, encoder=True, approx=False):
    """
    Calculate the number of parameters in Transformer Encoder/Decoder.
    Formulas are the following:
        multi-head attention: 4*(d_model^2 + d_model)
            if approx=False, 4*d_model^2 otherwise
        feed-forward: 2*d_model*d_ff + d_model + d_ff 
            if approx=False, 2*d_model*d_ff otherwise
        layer normalization: 2*d_model if approx=False, 0 otherwise
    Encoder block consists of: 
        1 multi-head attention block, 
        1 feed-forward net, and 
        2 layer normalizations.
    Decoder block consists of: 
        2 multi-head attention blocks, 
        1 feed-forward net, and 
        3 layer normalizations.
    :param d_model: (int) model dimensionality
    :param d_ff: (int) internal dimensionality of a feed-forward neural network
    :param encoder: (bool) if True, return the number of parameters of the Encoder, 
        otherwise the Decoder
    :param approx: (bool) if True, result is approximate (see formulas)
    :return: (int) number of learnable parameters in Transformer Encoder/Decoder
    """

    attention = 4 * (d_model ** 2 + d_model) if not approx else 4 * d_model ** 2
    feed_forward = 2 * d_model * d_ff + d_model + d_ff if not approx else 2 * d_model * d_ff
    layer_norm = 2 * d_model if not approx else 0

    return layers * (attention + feed_forward + 2 * layer_norm \
        if encoder else 2 * attention + feed_forward + 3 * layer_norm)

if __name__ == '__main__':
    model_params = transformer_count_params(d_model=2048, d_ff=2048 * 4, layers=24, encoder=True, approx=False)
    print(model_params * 2 / 1024 ** 2)