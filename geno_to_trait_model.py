# Define the autoencoder model
@tf.keras.utils.register_keras_serializable()
class trait_pred(Model):
    def __init__(self, width=5, depth=5, latent_dim = 100, **kwargs):
        self.latent_dim = latent_dim
        super().__init__(**kwargs)
        self.blocks = []
        self.depth = depth
        self.width = width
        self.fc_reg = keras.regularizers.L2(2e-2)
        # self.f_drop = feature_drop_layer(0.1, feature_dim = 1)
        self.drop = tf.keras.layers.Dropout(0.4)
        self.sd_noise = layers.GaussianNoise(1.0)
        self.mean_noise = layers.GaussianNoise(1.0)
        num_conv_iterations = 5
        self.conv_layers = []
        self.bnorm_a = layers.BatchNormalization()
        self.bnorm_b = layers.BatchNormalization()
        self.act_layer = layers.LeakyReLU(alpha=0.1)
        self.pool_step = tf.keras.layers.MaxPooling1D(
            pool_size=2, data_format="channels_first"
        )
        for i in range(num_conv_iterations):
            filter_size = 20 * (i + 1)
            self.conv_layers.append(
                tf.keras.layers.Conv1D(
                    filter_size,
                    kernel_size=2,
                    activation=None,
                    name="tunk_conv_" + str(i),
                    data_format="channels_first",
                    kernel_initializer="glorot_uniform",
                    kernel_constraint=keras.constraints.max_norm(2.0),
                )
            )
        for cur_width in range(self.width):
            cur_sub_block = []
            for layer_id in range(self.depth, 1, -1):
                cur_sub_block.append(
                    layers.Dense(
                        units=max(20, (layer_id) * 20),
                        kernel_regularizer=self.fc_reg,
                        kernel_initializer="glorot_uniform",
                        kernel_constraint=keras.constraints.max_norm(2.0),
                        name=f"trunk_dense_d_{layer_id}_w_{cur_width}",
                    )
                )

            cur_sub_block.append(
                layers.Dense(
                    units=self.latent_dim // self.width,
                    kernel_regularizer=self.fc_reg,
                    kernel_initializer="glorot_uniform",
                    kernel_constraint=keras.constraints.max_norm(2.0),
                    name=f"trunk_dense_final_w_{cur_width}",
                )
            )
            # cur_sub_block.append(self.act_layer)
            self.blocks.append(cur_sub_block)
        self.sd_layer = layers.Dense(
            units=6,
            kernel_regularizer=self.fc_reg,
            name="sd_layer",
            kernel_initializer="glorot_uniform",
            kernel_constraint=keras.constraints.max_norm(2.0),
        )  #
        self.sd_layer1 = layers.Dense(
            units=1,
            kernel_regularizer=self.fc_reg,
            name="final_sd_layer",
            kernel_initializer="glorot_uniform",
            kernel_constraint=keras.constraints.max_norm(2.0),
        )  #
        self.sd_bias_layer = layers.Dense(
            units=1,
            kernel_regularizer=self.fc_reg,
            name="sd_bias_layer",
            kernel_initializer=tf.keras.initializers.Zeros(),
            kernel_constraint=keras.constraints.max_norm(2.0),
        )  #
        self.mean_layer = layers.Dense(
            units=6,
            kernel_regularizer=self.fc_reg,
            name="mean_layer",
            kernel_initializer="glorot_uniform",
            kernel_constraint=keras.constraints.max_norm(2.0),
        )  #
        self.mean_layer1 = layers.Dense(
            units=1,
            kernel_regularizer=self.fc_reg,
            name="final_mean_layer",
            kernel_initializer="glorot_uniform",
            kernel_constraint=keras.constraints.max_norm(2.0),
        )  #
        self.mean_bias_layer = layers.Dense(
            units=1,
            kernel_regularizer=self.fc_reg,
            name="mean_bias_layer",
            kernel_constraint=keras.constraints.max_norm(2.0),
            kernel_initializer=tf.keras.initializers.Zeros(),
        )  #
        self.embedding = layers.Embedding(11, 4, name="ini_embedding")

        self.ini_conv = layers.Conv2D(
            64,
            kernel_size=[2, 3],
            activation=self.act_layer,
            name="gate_conv_" + str(0),
            data_format="channels_last",
        )
        # self.geno_sum_layer = Dense2dLayer(3, 1,
        #                         initializer = tf.keras.initializers.Constant(value=-1.0),
        #                         name = "test", activation = tf.keras.activations.linear) #activation = self.act_layer,
        self.geno_sum_layer = layers.Dense(self.latent_dim, activation=tf.keras.activations.linear)
        self.gate_conv_layers = []
        for i in range(5):
            filter_size = 20 * (i + 1)
            cur_conv_channel = "channels_last"
            self.gate_conv_layers.append(
                layers.Conv1D(
                    filter_size,
                    kernel_size=3,
                    activation=self.act_layer,
                    data_format=cur_conv_channel,
                    name=f"gate_conv_{i}",
                )
            )
            self.gate_conv_layers.append(
                layers.AveragePooling1D(pool_size=3, data_format=cur_conv_channel)
            )

    def get_config(self):
        config = super(trait_pred, self).get_config()
        config.update({"width": self.width, "depth": self.depth})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            **config
        )  # Use variable arguments to simplify reconstructing the object

    def parents_to_gate(self, parent_genos, training=False):
        parent_genos = tf.split(parent_genos, parent_genos.shape[1], axis=1)
        geno_out = []
        for cur_geno in parent_genos:
            cur_geno = tf.squeeze(cur_geno)
            for cur_conv in self.gate_conv_layers:
                cur_geno = cur_conv(cur_geno, training=training)
            geno_out.append(layers.Flatten()(cur_geno))
        geno_out = tf.concat(geno_out, axis=1)
        print("parent_genos.shape: ", geno_out.shape)
        parent_genos = tf.squeeze(self.geno_sum_layer(geno_out, training=training))
        # parent_genos = parent_genos - tf.reduce_min(parent_genos, axis = 0)
        # parent_genos = parent_genos / tf.reduce_max(parent_genos, axis = 0)
        return parent_genos  # tf.math.sigmoid(parent_genos)

    def call(
        self,
        parent_phenos,
        ini_embed_x,
        parent_genos,
        training=False,
        return_weights=False,
        return_activations=False,
    ):
        act_tracker = {}
        parent_genos = self.embedding(parent_genos, training=training)
        gate = self.parents_to_gate(parent_genos, training=training)
        act_tracker["gate"] = tf.reduce_mean(
            tf.reshape(gate, [gate.shape[0], -1]), axis=1
        )
        # embed_x = self.drop(embed_x * gate, training = training)
        embed_x = tf.concat([ini_embed_x, gate, ini_embed_x * gate], axis=1)
        # embed_x = [ini_embed_x, gate, ini_embed_x * gate]
        # embed_x = [tf.expand_dims(cur_embed, axis = 1) for cur_embed in embed_x]
        # # embed_x = tf.expand_dims(
        # #     ini_embed_x,
        # #     axis = 1)
        # conv_out = []
        # for cur_embed in embed_x:
        #     for cur_conv in self.conv_layers:
        #         cur_embed = cur_conv(cur_embed, training = training)
        #         cur_embed = self.act_layer(cur_embed)
        #         cur_embed = self.pool_step(cur_embed, training = training)
        #         act_tracker[cur_conv.name] = tf.reduce_mean(tf.reshape(cur_embed, [cur_embed.shape[0], -1]), axis = 1)
        #     conv_out.append(cur_embed)
        # embed_x = [tf.squeeze(self.pool_step(cur_embed)) for cur_embed in conv_out]
        # # embed_x = tf.squeeze(embed_x)
        # # embed_x = self.bnorm_a(embed_x, training = training)
        # embed_x = tf.concat(embed_x, axis = 1)
        outputs = []
        for cur_block in self.blocks:
            sub_x = self.drop(embed_x, training=training)
            for cur_layer in cur_block:
                # sub_x = self.drop(sub_x, training = training)
                sub_x = cur_layer(sub_x, training=training)
                sub_x = self.act_layer(sub_x)
                act_tracker[cur_layer.name] = tf.reduce_mean(
                    tf.reshape(sub_x, [sub_x.shape[0], -1]), axis=1
                )
            outputs.append(sub_x)
        block_out = tf.concat(outputs, axis=1) + ini_embed_x

        p_sd = tf.cast(parent_phenos[:, :, 1], dtype=block_out.dtype)
        p_mean = tf.cast(parent_phenos[:, :, 0], dtype=block_out.dtype)

        print("block_out.shape: ", block_out.shape)
        act_tracker["block_out"] = tf.reduce_mean(
            tf.reshape(sub_x, [sub_x.shape[0], -1]), axis=1
        )
        # sd_weights = self.sd_layer(
        #        self.drop(block_out, training = training),
        #        training = training)
        # sd_weights = self.act_layer(sd_weights)
        sd_weights = block_out  # tf.concat([block_out, p_sd], axis = 1)
        act_tracker["sd_weights"] = tf.reduce_mean(
            tf.reshape(sd_weights, [sd_weights.shape[0], -1]), axis=1
        )
        sd_scalings = self.sd_layer1(
            self.drop(block_out, training=training), training=training
        )
        act_tracker["sd_scalings"] = tf.reduce_mean(
            tf.reshape(sd_scalings, [sd_scalings.shape[0], -1]), axis=1
        )
        sd_scaled_parents = self.sd_bias_layer(
            self.sd_noise(p_sd, training=training), training=training
        )
        act_tracker["sd_scaled_parents"] = tf.reduce_mean(
            tf.reshape(sd_scaled_parents, [sd_scaled_parents.shape[0], -1]), axis=1
        )
        # sd_bias = self.sd_bias_layer(
        #    self.drop(sd_weights, training = training),
        #    training = training)
        # act_tracker["sd_bias"] = tf.reduce_mean(tf.reshape(sd_bias, [sd_bias.shape[0], -1]), axis = 1)

        # mean_weights = self.mean_layer(
        #    block_out,
        #    training = training)
        # mean_weights = self.act_layer(mean_weights)
        mean_weights = block_out  # tf.concat([mean_weights, p_mean], axis = 1)
        act_tracker["mean_weights"] = tf.reduce_mean(
            tf.reshape(mean_weights, [mean_weights.shape[0], -1]), axis=1
        )
        mean_scalings = self.mean_layer1(mean_weights, training=training)
        act_tracker["mean_scalings"] = tf.reduce_mean(
            tf.reshape(mean_scalings, [mean_scalings.shape[0], -1]), axis=1
        )
        mean_scaled_parents = self.mean_bias_layer(
            self.mean_noise(p_mean, training=training), training=training
        )
        act_tracker["mean_scaled_parents"] = tf.reduce_mean(
            tf.reshape(mean_scaled_parents, [mean_scaled_parents.shape[0], -1]), axis=1
        )

        scaled_mean = (
            mean_scalings + mean_scaled_parents
        )  # tf.cast(p_mean, dtype = mean_scalings.dtype)
        # scaled_mean = tf.reduce_sum(scaled_mean, axis = -1, keepdims = True)
        scaled_sd = (
            sd_scalings + sd_scaled_parents
        )  # tf.cast(p_sd, dtype = sd_scalings.dtype)
        # scaled_sd = tf.reduce_sum(scaled_sd, axis = -1, keepdims = True)
        trait_pred = tf.concat([scaled_mean, scaled_sd], axis=1)

        if return_weights:
            return trait_pred, scaling_weights, bias, gate
        if return_activations:
            return trait_pred, act_tracker, gate
        return trait_pred, {}, gate
