import tensorflow as tf
import numpy as np
import os


def residual_block(x, filters, kernel_size=3, stride=1, conv_shortcut=False, name=None):
    """A residual block.
    Arguments:
      x: input tensor.
      filters: integer, filters of the bottleneck layer.
      kernel_size: default 3, kernel size of the bottleneck layer.
      stride: default 1, stride of the first layer.
      conv_shortcut: default False, use convolution shortcut if True,
        otherwise identity shortcut.
      name: string, block label.
    Returns:
    Output tensor for the residual block.
    """
    bn_axis = -1  # ON the filters axis : [batch_size, seq_len, num_filters]
    momentum = 0.9
    preact = tf.keras.layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_preact_bn', momentum=momentum)(x)
    preact = tf.keras.layers.Activation('relu', name=name + '_preact_relu')(preact)

    if conv_shortcut:
        shortcut = tf.keras.layers.Conv1D(
            4 * filters, 1, strides=stride, name=name + '_0_conv')(preact)
    else:
        shortcut = tf.keras.layers.MaxPooling1D(1, strides=stride)(x) if stride > 1 else x

    x = tf.keras.layers.Conv1D(
      filters, 1, strides=1, use_bias=False, name=name + '_1_conv')(preact)
    x = tf.keras.layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn', momentum=momentum)(x)
    x = tf.keras.layers.Activation('relu', name=name + '_1_relu')(x)

    x = tf.keras.layers.ZeroPadding1D(padding=(1, 1), name=name + '_2_pad')(x)
    x = tf.keras.layers.Conv1D(
      filters,
      kernel_size,
      strides=stride,
      use_bias=False,
      name=name + '_2_conv')(x)
    x = tf.keras.layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn', momentum=momentum)(x)
    x = tf.keras.layers.Activation('relu', name=name + '_2_relu')(x)

    x = tf.keras.layers.Conv1D(4 * filters, 1, name=name + '_3_conv')(x)
    x = tf.keras.layers.Add(name=name + '_out')([shortcut, x])
    return x


def stack2(x, filters, blocks, stride1=2, name=None):
  """A set of stacked residual blocks.
  Arguments:
      x: input tensor.
      filters: integer, filters of the bottleneck layer in a block.
      blocks: integer, blocks in the stacked blocks.
      stride1: default 2, stride of the first layer in the first block.
      name: string, stack label.
  Returns:
      Output tensor for the stacked blocks.
  """
  x = residual_block(x, filters, conv_shortcut=True, name=name + '_block1')
  for i in range(2, blocks):
    x = residual_block(x, filters, name=name + '_block' + str(i))
  x = residual_block(x, filters, stride=stride1, name=name + '_block' + str(blocks))
  return x


def ResNet(stack_fn,
           preact,
           use_bias,
           model_name='resnet',
           include_top=True,
           weights='imagenet',
           input_tensor=None,
           input_shape=None,
           pooling=None,
           classes=1000,
           classifier_activation='softmax',
           **kwargs):
    """Instantiates the ResNet, ResNetV2, and ResNeXt architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    Caution: Be sure to properly pre-process your inputs to the application.
    Please see `applications.resnet.preprocess_input` for an example.
    Arguments:
    stack_fn: a function that returns output tensor for the
      stacked residual blocks.
    preact: whether to use pre-activation or not
      (True for ResNetV2, False for ResNet and ResNeXt).
    use_bias: whether to use biases for convolutional layers or not
      (True for ResNet and ResNetV2, False for ResNeXt).
    model_name: string, model name.
    include_top: whether to include the fully-connected
      layer at the top of the network.
    weights: one of `None` (random initialization),
      'imagenet' (pre-training on ImageNet),
      or the path to the weights file to be loaded.
    input_tensor: optional Keras tensor
      (i.e. output of `layers.Input()`)
      to use as image input for the model.
    input_shape: optional shape tuple, only to be specified
      if `include_top` is False (otherwise the input shape
      has to be `(224, 224, 3)` (with `channels_last` data format)
      or `(3, 224, 224)` (with `channels_first` data format).
      It should have exactly 3 inputs channels.
    pooling: optional pooling mode for feature extraction
      when `include_top` is `False`.
      - `None` means that the output of the model will be
          the 4D tensor output of the
          last convolutional layer.
      - `avg` means that global average pooling
          will be applied to the output of the
          last convolutional layer, and thus
          the output of the model will be a 2D tensor.
      - `max` means that global max pooling will
          be applied.
    classes: optional number of classes to classify images
      into, only to be specified if `include_top` is True, and
      if no `weights` argument is specified.
    classifier_activation: A `str` or callable. The activation function to use
      on the "top" layer. Ignored unless `include_top=True`. Set
      `classifier_activation=None` to return the logits of the "top" layer.
    **kwargs: For backwards compatibility only.
    Returns:
    A `keras.Model` instance.
    Raises:
    ValueError: in case of invalid argument for `weights`,
      or invalid input shape.
    ValueError: if `classifier_activation` is not `softmax` or `None` when
      using a pretrained top layer.
    """

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    # input_shape = imagenet_utils.obtain_input_shape(
    #     input_shape,
    #     default_size=224,
    #     min_size=32,
    #     data_format=backend.image_data_format(),
    #     require_flatten=include_top,
    #     weights=weights)
    momentum = 0.9
    if input_tensor is None:
        ecg_input = tf.keras.layers.Input(shape=input_shape)
    else:
        ecg_input = input_tensor

    bn_axis = -1

    x = tf.keras.layers.ZeroPadding1D(
      padding=(3, 3), name='conv1_pad')(ecg_input)
    x = tf.keras.layers.Conv1D(64, 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)

    if not preact:
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='conv1_bn', momentum=momentum)(x)
        x = tf.keras.layers.Activation('relu', name='conv1_relu')(x)

    x = tf.keras.layers.ZeroPadding1D(padding=(1, 1), name='pool1_pad')(x)
    x = tf.keras.layers.MaxPooling1D(3, strides=2, name='pool1_pool')(x)

    x = stack_fn(x)

    if preact:
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='post_bn', momentum=momentum)(x)
        x = tf.keras.layers.Activation('relu', name='post_relu')(x)

    if include_top:
        x = tf.keras.layers.GlobalAveragePooling1D(name='avg_pool')(x)
        # imagenet_utils.validate_activation(classifier_activation, weights)
        x = tf.keras.layers.Dense(classes, activation=classifier_activation,
                         name='predictions')(x)
    else:
        if pooling == 'avg':
            x = tf.keras.layers.GlobalAveragePooling1D(name='avg_pool')(x)
        elif pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling1D(name='max_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    # if input_tensor is not None:
    #   inputs = layer_utils.get_source_inputs(input_tensor)
    # else:
    #   inputs = img_input

    # Create model.
    # model = training.Model(inputs, x, name=model_name)
    #
    # # Load weights.
    # if (weights == 'imagenet') and (model_name in WEIGHTS_HASHES):
    #   if include_top:
    #     file_name = model_name + '_weights_tf_dim_ordering_tf_kernels.h5'
    #     file_hash = WEIGHTS_HASHES[model_name][0]
    #   else:
    #     file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_notop.h5'
    #     file_hash = WEIGHTS_HASHES[model_name][1]
    #   weights_path = data_utils.get_file(
    #       file_name,
    #       BASE_WEIGHTS_PATH + file_name,
    #       cache_subdir='models',
    #       file_hash=file_hash)
    #   model.load_weights(weights_path)
    # elif weights is not None:
    #   model.load_weights(weights)
    model = tf.keras.Model(ecg_input, x, name=model_name)
    return model


def ResNet50V2(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax',
):
    """Instantiates the ResNet50V2 architecture."""
    def stack_fn(x):
        x = stack2(x, 64, 3, name='conv2')
        x = stack2(x, 128, 4, name='conv3')
        x = stack2(x, 256, 6, name='conv4')
        return stack2(x, 512, 3, stride1=1, name='conv5')

    return ResNet(
          stack_fn,
          True,
          True,
          'resnet50v2',
          include_top,
          weights,
          input_tensor,
          input_shape,
          pooling,
          classes,
          classifier_activation=classifier_activation,
      )


# m = ResNet50V2(
#     include_top=True,
#     weights=None,
#     input_tensor=None,
#     input_shape=(500, 12),
#     pooling=None,
#     classes=2,
#     classifier_activation='softmax')
#
# inp = tf.convert_to_tensor(np.zeros((2, 500, 12)), dtype=tf.float64)
#
# out = m(inp)
# print(out.shape)