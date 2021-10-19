import tensorflow as tf
import math
from tensorflow.keras import backend as K

def round_filters(filters, multiplier):
    depth_divisor = 8
    min_depth = None
    min_depth = min_depth or depth_divisor
    filters = filters * multiplier
    new_filters = max(min_depth, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)

def round_repeats(repeats, multiplier):
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))

class SEBlock(tf.keras.layers.Layer):
    def __init__(self, input_channels, ratio=0.25):
        super(SEBlock, self).__init__()
        self.num_reduced_filters = max(1, int(input_channels * ratio))
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.reduce_conv = tf.keras.layers.Conv2D(filters=self.num_reduced_filters,
                                                  kernel_size=(1, 1),
                                                  strides=1,
                                                  padding="same")
        self.expand_conv = tf.keras.layers.Conv2D(filters=input_channels,
                                                  kernel_size=(1, 1),
                                                  strides=1,
                                                  padding="same")

    def get_config(self):
        return {'num_reduced_filters': self.num_reduced_filters}

    def call(self, inputs, **kwargs):
        branch = self.pool(inputs)
        branch = tf.expand_dims(input=branch, axis=1)
        branch = tf.expand_dims(input=branch, axis=1)
        branch = self.reduce_conv(branch)
        branch = tf.nn.swish(branch)
        branch = self.expand_conv(branch)
        branch = tf.nn.sigmoid(branch)
        output = inputs * branch
        return output

class MBConv(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, expansion_factor, stride, k, drop_connect_rate):
        super(MBConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.drop_connect_rate = drop_connect_rate
        self.conv1 = tf.keras.layers.Conv2D(filters=in_channels * expansion_factor,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same",
                                            use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dwconv = tf.keras.layers.DepthwiseConv2D(kernel_size=(k, k),
                                                      strides=stride,
                                                      padding="same",
                                                      use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.se = SEBlock(input_channels=in_channels * expansion_factor)
        self.conv2 = tf.keras.layers.Conv2D(filters=out_channels,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same",
                                            use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(rate=drop_connect_rate)

    def get_config(self):
        return {'in_channels': self.in_channels, 'out_channels': self.out_channels,
        'stride': self.stride, 'drop_connect_rate': self.drop_connect_rate}

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.swish(x)
        x = self.dwconv(x)
        x = self.bn2(x, training=training)
        x = self.se(x)
        x = tf.nn.swish(x)
        x = self.conv2(x)
        x = self.bn3(x, training=training)
        if self.stride == 1 and self.in_channels == self.out_channels:
            if self.drop_connect_rate:
                x = self.dropout(x, training=training)
            x = tf.keras.layers.add([x, inputs])
        return x

def build_mbconv_block(in_channels, out_channels, layers, stride, expansion_factor, k, drop_connect_rate):
    block = tf.keras.Sequential()
    for i in range(layers):
        if i == 0:
            block.add(MBConv(in_channels=in_channels,
                             out_channels=out_channels,
                             expansion_factor=expansion_factor,
                             stride=stride,
                             k=k,
                             drop_connect_rate=drop_connect_rate))
        else:
            block.add(MBConv(in_channels=out_channels,
                             out_channels=out_channels,
                             expansion_factor=expansion_factor,
                             stride=1,
                             k=k,
                             drop_connect_rate=drop_connect_rate))
    return block

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, shape_input):
        super(AttentionLayer, self).__init__()

        self.shape = shape_input
        self.resize = tf.keras.layers.experimental.preprocessing.Resizing(shape_input[1], shape_input[2])
        self.conv1 = tf.keras.layers.Conv2D(self.shape[-1], kernel_size=(1,1), strides=1, padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(self.shape[-1], kernel_size=(1,1), strides=1, padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(self.shape[-1], kernel_size=(1,1), strides=1, padding="same")
        self.bn3 = tf.keras.layers.BatchNormalization()
      
    def build(self, input_shape):
        self.alpha = tf.Variable(name="alpha", initial_value=1.0, trainable=True)

    def call(self, inputs, training=None, **kwargs):
        x = inputs[0]

        rs = self.resize(inputs[1])
        rs = tf.where(rs >= 0.5, 1.0, 0.0)
        rs = self.conv1(rs)
        rs = self.bn1(rs, training=training)
        rs = self.conv2(rs)
        rs = self.bn2(rs, training=training)
        rs = self.conv3(rs)
        rs = self.bn3(rs, training=training)

        att = tf.nn.softmax(rs)
        att = x * att
        att = att * self.alpha

        output = x + rs + att
        return output

    def get_config(self):
        return {'shape': self.shape}

def teste_eff_mask(width_coefficient, depth_coefficient, dropout_rate, drop_connect_rate=0.2):
    input = tf.keras.layers.Input(shape=(224, 224, 3))
    mask = tf.keras.layers.Input(shape=(224, 224, 1))

    x = tf.keras.layers.Conv2D(filters=round_filters(32, width_coefficient), 
                                kernel_size=(3, 3), strides=2, padding="same",
                                use_bias=False)(input)
    x = tf.keras.layers.BatchNormalization()(x)

    x = AttentionLayer(x.shape)([x, mask])

    # bloco 1
    x = build_mbconv_block(in_channels=round_filters(32, width_coefficient),
                            out_channels=round_filters(16, width_coefficient),
                            layers=round_repeats(1, depth_coefficient), stride=1,
                            expansion_factor=1, k=3, drop_connect_rate=drop_connect_rate)(x)

    # bloco 2
    x = build_mbconv_block(in_channels=round_filters(16, width_coefficient),
                            out_channels=round_filters(24, width_coefficient),
                            layers=round_repeats(2, depth_coefficient), stride=2,
                            expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate)(x)

    # bloco 3
    x = build_mbconv_block(in_channels=round_filters(24, width_coefficient),
                            out_channels=round_filters(40, width_coefficient),
                            layers=round_repeats(2, depth_coefficient), stride=2,
                            expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate)(x)

    # bloco 4                         
    x = build_mbconv_block(in_channels=round_filters(40, width_coefficient),
                            out_channels=round_filters(80, width_coefficient),
                            layers=round_repeats(3, depth_coefficient), stride=2,
                            expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate)(x)

    x = AttentionLayer(x.shape)([x, mask])

    # bloco 5                         
    x = build_mbconv_block(in_channels=round_filters(80, width_coefficient),
                            out_channels=round_filters(112, width_coefficient),
                            layers=round_repeats(3, depth_coefficient), stride=1,
                            expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate)(x)

    # bloco 6
    x = build_mbconv_block(in_channels=round_filters(112, width_coefficient),
                            out_channels=round_filters(192, width_coefficient),
                            layers=round_repeats(4, depth_coefficient), stride=2,
                            expansion_factor=6, k=5, drop_connect_rate=drop_connect_rate)(x)

    # bloco 7
    x = build_mbconv_block(in_channels=round_filters(192, width_coefficient),
                            out_channels=round_filters(320, width_coefficient),
                            layers=round_repeats(1, depth_coefficient), stride=1,
                            expansion_factor=6, k=3, drop_connect_rate=drop_connect_rate)(x)

    x = tf.keras.layers.Conv2D(filters=round_filters(1280, width_coefficient),
                                        kernel_size=(1, 1),
                                        strides=1,
                                        padding="same",
                                        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = AttentionLayer(x.shape)([x, mask])

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inputs=[input, mask], outputs=[x])

def get_att_eff_b0():
    return teste_eff_mask(1.0, 1.0, 0.2, 0.2)
