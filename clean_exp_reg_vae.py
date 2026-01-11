# Define the autoencoder model
@tf.keras.utils.register_keras_serializable()
class reg_vae(Model):
    def __init__(
        self, latent_dim, encoder_width=3, decoder_depth=1, pheno_only=False, **kwargs
    ):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.encoder_width = encoder_width
        self.decoder_depth = decoder_depth
        self.p_encoder = p_encoder(self.latent_dim)
        self.encoder = encoder(latent_dim, self.encoder_width)
        self.decoder = decoder(latent_dim, self.decoder_depth)
        self.regressor = trait_pred(width=5, depth=1, latent_dim=self.latent_dim)
        self.pheno_only = pheno_only
        self.loss_fn = elbo_loss()
        self.pop_shared_genos = embed_pop_distances
        self.rec_loss_tracker = tf.keras.metrics.Mean(
            name="rec_loss"
        )
        self.p_rec_loss_tracker = tf.keras.metrics.Mean(
            name="p_rec_loss"
        )
        self.reg_loss_tracker = tf.keras.metrics.Mean(name="reg_loss")
        self.p_reg_loss_tracker = tf.keras.metrics.Mean(name="p_reg_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.p_kl_loss_tracker = tf.keras.metrics.Mean(name="p_kl_loss")
        self.kl_scale_tracker = tf.keras.metrics.Mean(name="kl_scale")
        self.pop_shared_genos_tracker = tf.keras.metrics.Mean(name="pop_shared_genos")
        self.trait_diff_tracker = tf.keras.metrics.Mean(name="trait_diff")
        self.mean_loss_tracker = tf.keras.metrics.MeanAbsolutePercentageError(
            name="mean_deviation"
        )
        self.sd_loss_tracker = tf.keras.metrics.MeanAbsolutePercentageError(
            name="sd_deviation"
        )
        self.epoch_tracker = tf.keras.metrics.Mean(name="epoch")
        self.cat_acc_tracker = tf.keras.metrics.SparseCategoricalAccuracy(
            name="cat_acc"
        )
        self.p_cat_acc_tracker = tf.keras.metrics.SparseCategoricalAccuracy(
            name="p_cat_acc"
        )
        self.class_acc_tracker = [
            tf.keras.metrics.SparseCategoricalAccuracy(name=f"{cur_class}_acc")
            for cur_class in range(11)
        ]
        self.reg_opt = optimizers.AdamW(learning_rate=1e-3, clipnorm=1.0)
        self.dec_opt = optimizers.AdamW(learning_rate=1e-3)
        self.enc_opt = optimizers.AdamW(learning_rate=1e-4)
        self.p_enc_opt = optimizers.AdamW(learning_rate=1e-4)
        self.prev_reg_loss = tf.Variable(initial_value=1.0, trainable=False)
        self.prev_rec_loss = tf.Variable(initial_value=1.0, trainable=False)
        self.train_cycle_count = tf.Variable(initial_value=0.0, trainable=False)
        self.single_opt_cycle_count = tf.Variable(initial_value=0.0, trainable=False)
        self.all_opt_cycle_count = tf.Variable(initial_value=0.0, trainable=False)

    def get_config(self):
        config = super(reg_vae, self).get_config()
        config.update(
            {
                "latent_dim": self.latent_dim,
                "encoder_width": self.encoder_width,
                "decoder_depth": self.decoder_depth,
                "pheno_only": self.pheno_only,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def sample_z(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    @tf.function
    def call(self, data, training=False, return_activations=False):
        geno_x, trait_x, meta_x = data
        parent_trait = trait_x
        parents_genos = geno_x[:, :2, ...]
        _, _, enc_act = self.encoder.call(
            geno_x, meta_x, training=training, return_activations=return_activations
        )
        # logvar = tf.constant(0.0)
        mean, logvar, _ = self.p_encoder.call(
            parents_genos, meta_x, training=training, return_activations=return_activations
        )
        embed_x = self.sample_z(mean, logvar)
        geno_logits, dec_act, dec_gate, geno_pred = self.decoder.call(
            parents_genos,
            embed_x,
            meta_x,
            training=training,
            return_activations=return_activations,
        )
        pheno_pred, reg_act, reg_gate = self.regressor.call(
            parent_trait,
            embed_x,
            parents_genos,
            training=training,
            return_activations=return_activations,
        )
        all_activations = {"encoder": enc_act, "decoder": dec_act, "regressor": reg_act}
        if self.pheno_only:
            return pheno_pred
        return (
            embed_x,
            geno_logits,
            pheno_pred,
            mean,
            logvar,
            dec_gate,
            reg_gate,
            all_activations,
            geno_pred,
        )

    @tf.function
    def update_trackers(
        self,
        rec_loss,
        p_rec_loss,
        reg_loss,
        p_reg_loss,
        kl_loss,
        p_kl_loss,
        kl_scale,
        geno_labels,
        geno_logits,
        p_geno_logits,
        cur_epoch,
        trait_pred,
        trait_true,
        pop_shared_genos,
        trait_diff
    ):
        self.rec_loss_tracker.update_state(rec_loss)
        self.p_rec_loss_tracker.update_state(p_rec_loss)
        self.cat_acc_tracker.update_state(geno_labels, geno_logits)
        self.p_cat_acc_tracker.update_state(geno_labels, p_geno_logits)
        self.kl_scale_tracker.update_state(kl_scale)
        self.kl_loss_tracker.update_state(kl_loss)
        self.p_kl_loss_tracker.update_state(p_kl_loss)
        self.epoch_tracker.update_state(cur_epoch)
        self.reg_loss_tracker.update_state(reg_loss)
        self.p_reg_loss_tracker.update_state(p_reg_loss)
        self.pop_shared_genos_tracker.update_state(pop_shared_genos)
        self.trait_diff_tracker.update_state(trait_diff)
        self.mean_loss_tracker.update_state(trait_true[:, 0], trait_pred[:, 0])
        self.sd_loss_tracker.update_state(trait_true[:, 1], trait_pred[:, 1])

        for class_id in range(len(self.class_acc_tracker)):
            class_mask = tf.equal(geno_labels, class_id)
            class_y_truth = tf.boolean_mask(geno_labels, class_mask)
            class_y_pred = tf.boolean_mask(
                tf.nn.softmax(geno_logits, axis=-1), class_mask
            )
            self.class_acc_tracker[class_id].update_state(class_y_truth, class_y_pred)

        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def train_step(self, data, cur_epoch, return_activations=False):
        training = True
        geno_x, trait_x, meta_x = data
        seq_pos, chr_pos, pop_x = meta_x
        child_trait = trait_x[1]
        parent_trait = trait_x[0]
        parents_genos = geno_x[:, :2, ...]
        child_genos = geno_x[:, 2, ...]
        # kl_scale scaling up over epochs
        kl_scale = 0.1#tf.maximum(1.0, (tf.cast(cur_epoch, tf.float32) + 100.0) / 100.0)
        #kl_scale = 10.0
        with tf.GradientTape(persistent=True) as grad_tape:
            p_mean, p_logvar, _ = self.p_encoder.call(
                parents_genos, meta_x, training=training, return_activations=return_activations
            )
            mean, logvar, enc_act = self.encoder.call(
                geno_x, meta_x, training=training, return_activations=return_activations
            )
            logvar = tf.clip_by_value(logvar, -7, 7)
            p_logvar = tf.clip_by_value(p_logvar, -7, 7)
            mean = tf.clip_by_value(mean, -30, 30)
            p_mean = tf.clip_by_value(p_mean, -30, 30)
            embed_x = self.sample_z(mean, logvar)
            p_embed_x = self.sample_z(p_mean, p_logvar)
            geno_logits, dec_act, dec_gate, geno_pred = self.decoder.call(
                parents_genos,
                embed_x,
                meta_x,
                training=training,
                return_activations=return_activations,
            )
            pheno_pred, reg_act, reg_gate = self.regressor.call(
                parent_trait,
                embed_x,
                parents_genos,
                training=training,
                return_activations=return_activations,
            )
            p_geno_logits, _, _, _ = self.decoder.call(
                parents_genos,
                p_embed_x,
                meta_x,
                training=training,
                return_activations=return_activations,
            )
            p_pheno_pred, _, _ = self.regressor.call(
                parent_trait,
                p_embed_x,
                parents_genos,
                training=training,
                return_activations=return_activations,
            )
            p_geno_off_tokens = tf.argmax(
                tf.stop_gradient(p_geno_logits),
                axis=-1
                )
            p_geno_off_tokens = tf.cast(p_geno_off_tokens, dtype=tf.float32)
            new_geno_x = tf.concat(
                [parents_genos, tf.expand_dims(p_geno_off_tokens, axis=1)],
                axis=1,
            )
            new_mean, new_logvar, enc_act = self.encoder.call(
                new_geno_x, meta_x, training=training, return_activations=return_activations
            )
            new_embed_x = self.sample_z(new_mean, new_logvar)
            new_geno_logits, dec_act, dec_gate, geno_pred = self.decoder.call(
                parents_genos,
                new_embed_x,
                meta_x,
                training=training,
                return_activations=return_activations,
            )

            new_pheno_pred, _, _ = self.regressor.call(
                parent_trait,
                new_embed_x,
                parents_genos,
                training=training,
                return_activations=return_activations,
            )
            kl_loss, p_kl_loss, reg_loss, rec_loss = self.loss_fn(
                child_genos,
                geno_logits,
                mean,
                logvar,
                trait_pred=pheno_pred,
                trait_true=child_trait,
                epoch=cur_epoch,
                p_mean=p_mean,
                p_logvar=p_logvar,
                training=training
            )
            _, _, p_reg_loss, p_rec_loss = self.loss_fn(
                p_geno_off_tokens,
                new_geno_logits,
                mean,
                logvar,
                trait_pred=new_pheno_pred,
                trait_true=p_pheno_pred,
                epoch=cur_epoch,
                p_mean=p_mean,
                p_logvar=p_logvar,
                training=training
            )
            rec_loss = rec_loss
            reg_loss = reg_loss
            total_loss_kl = kl_loss * kl_scale + reg_loss + rec_loss * 1000
            kl_forward = kl_divergence(p_mean, p_logvar, new_mean, new_logvar)
            kl_reverse = kl_divergence(new_mean, new_logvar, p_mean, p_logvar)
            cyclic_loss = 0.5 * (kl_forward + kl_reverse)
            total_p_loss = p_kl_loss * kl_scale + cyclic_loss * kl_scale + p_reg_loss + p_rec_loss * 1000
            pop_shared_genos = self.pop_shared_genos(geno_logits, pop_x)
            trait_diff = trait_pop_distances(pheno_pred, pop_x)

        self.prev_reg_loss.assign(reg_loss)
        self.prev_rec_loss.assign(rec_loss)
        self.train_cycle_count.assign_add(1.0)
        p_enc_grads = grad_tape.gradient(
            total_p_loss, self.p_encoder.trainable_variables
        )
        self.p_enc_opt.apply_gradients(
            zip(p_enc_grads, self.p_encoder.trainable_variables)
        )
        enc_grads = grad_tape.gradient(total_loss_kl, self.encoder.trainable_variables)
        self.enc_opt.apply_gradients(
            zip(enc_grads, self.encoder.trainable_variables))
        dec_grads = grad_tape.gradient(
            rec_loss, self.decoder.trainable_variables
        )
        self.dec_opt.apply_gradients(
            zip(dec_grads, self.decoder.trainable_variables)
        )
        reg_grads = grad_tape.gradient(
            reg_loss, self.regressor.trainable_variables
        )
        self.reg_opt.apply_gradients(
            zip(reg_grads, self.regressor.trainable_variables)
        )

        all_activations = {"encoder": enc_act, "decoder": dec_act, "regressor": reg_act}
        del grad_tape

        self.update_trackers(
            rec_loss,
            p_rec_loss,
            reg_loss,
            p_reg_loss,
            kl_loss,
            p_kl_loss,
            kl_scale,
            child_genos,
            geno_logits,
            new_geno_logits,
            cur_epoch,
            pheno_pred,
            child_trait,
            pop_shared_genos,
            trait_diff
        )
        return all_activations

    @tf.function
    def test_step(self, data, cur_epoch):
        training = False
        geno_x, trait_x, meta_x = data
        seq_pos, chr_pos, pop_x = meta_x
        child_trait = trait_x[1]
        parent_trait = trait_x[0]
        parents_genos = geno_x[:, :2, ...]
        child_genos = geno_x[:, 2, ...]
        c_mean, c_logvar, _ = self.encoder.call(
                geno_x, meta_x,
                training=training
        )
        mean, logvar, _ = self.p_encoder.call(
            parents_genos, meta_x, training=training
        )  # mean, logvar,
        # logvar = tf.constant(0.0)
        embed_x = self.sample_z(mean, logvar)
        c_embed_x = self.sample_z(c_mean, c_logvar)
        geno_logits, dec_act, dec_gate, geno_pred = self.decoder.call(
            parents_genos, embed_x, meta_x, training=training
        )
        c_geno_logits, _, _, c_geno_pred = self.decoder.call(
            parents_genos, c_embed_x, meta_x, training=training
        )
        c_pheno_pred, reg_act, reg_gate = self.regressor.call(
            parent_trait, c_embed_x, parents_genos, training=training
        )
        pheno_pred, reg_act, reg_gate = self.regressor.call(
            parent_trait, embed_x, parents_genos, training=training
        )        
        kl_loss, kl_loss, reg_loss, rec_loss = self.loss_fn(
            child_genos,
            geno_logits,
            mean,
            logvar,
            trait_pred=pheno_pred,
            trait_true=child_trait,
            epoch=cur_epoch,
            training=training
        )
        _, c_kl_loss, c_reg_loss, c_rec_loss = self.loss_fn(
            child_genos,
            c_geno_logits,
            mean,
            logvar,
            trait_pred=c_pheno_pred,
            trait_true=child_trait,
            epoch=cur_epoch,
            training=training
        )
        kl_scale = 1.0
        rec_loss = kl_loss * kl_scale + rec_loss
        reg_loss = kl_loss * kl_scale + reg_loss
        pop_shared_genos = self.pop_shared_genos(geno_logits, pop_x)
        trait_diff = trait_pop_distances(pheno_pred, pop_x)
        self.update_trackers(
            c_rec_loss,
            rec_loss,
            c_reg_loss,
            reg_loss,
            c_kl_loss,
            kl_loss,
            kl_scale,
            child_genos,
            c_geno_logits,
            geno_logits,
            cur_epoch,
            pheno_pred,
            child_trait,
            pop_shared_genos,
            trait_diff
        )
        return {}

    @property
    def metrics(self):
        return [
            self.kl_scale_tracker,
            self.reg_loss_tracker,
            self.p_reg_loss_tracker,
            self.epoch_tracker,
            self.cat_acc_tracker,
            self.p_cat_acc_tracker,
            self.mean_loss_tracker,
            self.sd_loss_tracker,
            self.kl_loss_tracker,
            self.p_kl_loss_tracker,
            self.pop_shared_genos_tracker,
            self.trait_diff_tracker,
            self.rec_loss_tracker,
            self.p_rec_loss_tracker
        ] + self.class_acc_tracker
