@tf.keras.utils.register_keras_serializable()
class decoder(Model):
    def __init__(self, latent_dim, depth = 1,**kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.depth = depth
        self.fc_reg = keras.regularizers.L2(6e-3)
        # self.drop = feature_drop_layer(0.2, feature_dim = 1)
        self.f_drop = feature_drop_layer(0.05, feature_dim = 1)
        self.drop = tf.keras.layers.Dropout(0.3)
        self.blocks = []
        self.act_layer = tf.keras.layers.LeakyReLU(alpha = 0.5)
        for cur_width in range(11):
            # Create a block of layers
            block = []
            for cur_depth in range(1, self.depth):
                block.append(layers.Dense(units=100 * cur_depth,
                    name = f"dec_dense_d_{cur_depth}_w_{cur_width}",
                    activation = tf.keras.layers.LeakyReLU(0.1)))
                # block.append(layers.BatchNormalization(center = True,
                #     name = f"dec_batch_norm_d_{cur_depth}_w_{cur_width}"))
                # block.append(self.drop)
            block.append(layers.Dense(units=1144, name = f"dec_final_dense_w_{cur_width}",
                                      activation = keras.activations.silu))
            self.blocks.append(block)
        # self.post_block_nn = Dense2dLayer_mod(1144, 1144, initializer = keras.initializers.GlorotNormal(),
        #                            activation = tf.keras.activations.linear, name = "test")
        # self.p_dense = layers.Dense(units = )
        self.ini_conv = layers.Conv2D(
                64,
                kernel_size=[2, 3],
                activation=self.act_layer,
                name="dec_ini_conv" + str(0),
                data_format = "channels_last"
            )

        self.geno_sum_layer = Dense2dLayer(3, 1, initializer = keras.initializers.GlorotNormal(),
                                   activation = self.act_layer, name = "test")
        self.diff_to_z_att = tf.keras.layers.MultiHeadAttention(
            num_heads=1,
            key_dim=11,  # Match embedding dim
            value_dim=11,
            dropout=0.0,
            name = "dec_diff_to_z_att"
        )
                                   
        self.gate_conv_layers = []
        for i in range(5):
            filter_size = 20 * (i + 1)
            cur_conv_channel = "channels_last"
            self.gate_conv_layers.append(
                layers.Conv1D(
                    filter_size,
                    kernel_size=3,
                    activation=self.act_layer,
                    data_format = cur_conv_channel,
                    name = f"dec_conv_{i}"
                )
            )
            self.gate_conv_layers.append(
                layers.AveragePooling1D(
                    pool_size=3,
                    data_format = cur_conv_channel,
                    name = f"dec_avg_pool_{i}"))
        self.embedding = layers.Embedding(11, 11, name = "dec_ini_embedding")
        self.chr_embedding = layers.Embedding(11, 11, name="dec_chr_embedding")
        self.p_embedding = layers.Embedding(221, 11, name="dec_ini_embedding")
        self.p_embedding_att = layers.Embedding(221, 11, name="dec_ini_embedding")
        self.e_final_dense = layers.EinsumDense("abc,cd->abd",
                                      output_shape=(None, 11),
                                      bias_axes="d")
        self.final_e_drop = tf.keras.layers.Dropout(0.4)
        self.p_final_dense = layers.EinsumDense("abc,cd->abd",
                              output_shape=(None, 11),
                              bias_axes="d")




    def get_config(self):
        config = super(decoder, self).get_config()
        config.update({
            'latent_dim': self.latent_dim,
            'depth': self.depth
        })
        return config


    @classmethod
    def from_config(cls, config):
        return cls(**config)
    def parents_to_gate(self, parent_genos, training = False):
        parent_genos = tf.squeeze(self.ini_conv(parent_genos, training = training))
        for cur_conv in self.gate_conv_layers:
            parent_genos = cur_conv(parent_genos)
        parent_genos = tf.squeeze(self.geno_sum_layer(parent_genos))
        return parent_genos
    
    def call(self, parents, embed_x, pos_data, training = False, return_activations = False):
        act_tracker = {}
        outputs = []
        # seq_pos, chr_pos, pop_x = pos_data
        # seq_pos = new_positional_encoding(seq_pos[0, :], 11)
        # chr_pos = self.chr_embedding(chr_pos, training=training)
        parent_split = tf.split(parents, num_or_size_splits =parents.shape[1], axis = 1)
        #parent_split = [tf.squeeze(x, axis = 1) for x in parent_split]
        parents_diff_ini = cantor_pairing(tf.concat([parent_split[0], parent_split[1]], axis = 1))
        parents_diff = self.p_embedding(parents_diff_ini, training = training)
        parents_diff_att = self.p_embedding_att(parents_diff_ini, training = training)
        parent_genos = self.embedding(parents, training = training)
        gate = self.parents_to_gate(parent_genos, training=training)
        act_tracker["gate"] = tf.reduce_mean(tf.reshape(gate, [gate.shape[0], -1]), axis = 1)
        embed_x = embed_x# * gate
        
        # embed_x = self.drop(embed_x, training = training)
        for block in self.blocks:
            sub_x = self.drop(embed_x, training=training)
            for layer in block:
                sub_x = layer(sub_x, training = training)
                # sub_x = self.act_layer(sub_x)
                act_tracker[layer.name] = tf.reduce_mean(tf.reshape(sub_x, [sub_x.shape[0], -1]), axis = 1)
            sub_x = tf.reshape(sub_x, (-1, 1144, 1))
            outputs.append(sub_x)
        geno_pred = tf.concat(outputs, axis=-1)

        # geno_pred = tf.nn.softmax(geno_pred)
        # prod_parents = tf.math.reduce_sum(parent_genos, axis = 1)
        # pred = tf.cast(prod_parents, dtype = geno_pred.dtype) + geno_pred
        pred = parents_diff + self.e_final_dense(
          geno_pred,
          training = training)
        if return_activations:
            return pred, act_tracker, gate
        else:
            return pred, {}, gate, geno_pred
