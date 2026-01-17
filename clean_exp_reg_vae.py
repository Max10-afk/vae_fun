import tensorflow as tf
from tensorflow.keras import Model, optimizers, metrics

@tf.keras.utils.register_keras_serializable()
class RegVAE(Model):
    def __init__(self, latent_dim, encoder_width=3, decoder_depth=1, pheno_only=False, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.encoder_width = encoder_width
        self.decoder_depth = decoder_depth
        self.pheno_only = pheno_only
        
        # --- Sub-models ---
        self.p_encoder = p_encoder(self.latent_dim)
        self.encoder = encoder(latent_dim, self.encoder_width)
        self.decoder = decoder(latent_dim, self.decoder_depth)
        self.regressor = trait_pred(width=5, depth=1, latent_dim=self.latent_dim)
        
        # --- Utilities ---
        self.loss_fn = elbo_loss()
        self.pop_shared_genos = embed_pop_distances
        
        # --- Optimizers ---
        self.opts = {
            'reg': optimizers.AdamW(learning_rate=1e-3, clipnorm=1.0),
            'dec': optimizers.AdamW(learning_rate=1e-3),
            'enc': optimizers.AdamW(learning_rate=1e-4),
            'p_enc': optimizers.AdamW(learning_rate=1e-4)
        }
        self.num_pops = 25

        # --- Metrics ---
        self._init_trackers()

    def _init_trackers(self):
        """Initializes all metrics in a loop."""
        tracker_names = [
            "rec_loss", "p_rec_loss", "reg_loss", "p_reg_loss", 
            "kl_loss", "p_kl_loss", "kl_scale", "pop_shared_genos", 
            "trait_diff", "epoch"
        ]
        self.trackers = {name: metrics.Mean(name=name) for name in tracker_names}
        self.trackers['mean_deviation'] = metrics.MeanAbsolutePercentageError(name="mean_deviation")
        self.trackers['sd_deviation'] = metrics.MeanAbsolutePercentageError(name="sd_deviation")
        self.trackers['cat_acc'] = metrics.SparseCategoricalAccuracy(name="cat_acc")
        self.trackers['p_cat_acc'] = metrics.SparseCategoricalAccuracy(name="p_cat_acc")
        self.class_acc_trackers = [metrics.SparseCategoricalAccuracy(name=f"{i}_acc") for i in range(11)]
        self.pop_trackers = {
            'pop_true_mean': PopWiseMean(self.num_pops, name="pop_true_mean"),
            'pop_pred_mean': PopWiseMean(self.num_pops, name="pop_pred_mean"),
            'pop_true_std':  PopWiseMean(self.num_pops, name="pop_true_std"),
            'pop_pred_std':  PopWiseMean(self.num_pops, name="pop_pred_std"),
        }

    def get_config(self):
        config = super().get_config()
        config.update({
            "latent_dim": self.latent_dim,
            "encoder_width": self.encoder_width,
            "decoder_depth": self.decoder_depth,
            "pheno_only": self.pheno_only,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def sample_z(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * 0.5) + mean

    def _parse_data(self, data):
        """Helper to unpack complex data tuples."""
        geno_x, trait_x, meta_x = data
        parents_genos = geno_x[:, :2, ...]
        child_genos = geno_x[:, 2, ...]
        parent_trait = trait_x[0]
        child_trait = trait_x[1]
        # seq_pos, chr_pos, pop_x = meta_x # Unpack if needed explicitly
        return geno_x, meta_x, parents_genos, child_genos, parent_trait, child_trait

    def _generate_cyclic_offspring(self, p_geno_logits, parents_genos, meta_x, training):
        """Generates new offspring based on parent logits (Cyclic Step)."""
        # Argmax is non-differentiable, stop gradient naturally applies here for the token indices
        p_geno_off_tokens = tf.argmax(p_geno_logits, axis=-1, output_type=tf.int32)
        p_geno_off_tokens = tf.cast(p_geno_off_tokens, dtype=tf.float32)
        
        new_geno_x = tf.concat([parents_genos, tf.expand_dims(p_geno_off_tokens, axis=1)], axis=1)
        
        new_mean, new_logvar, _ = self.encoder(new_geno_x, meta_x, training=training)
        new_embed_x = self.sample_z(new_mean, new_logvar)
        
        return new_geno_x, new_embed_x, new_mean, new_logvar

    @tf.function
    def train_step(self, data, cur_epoch=0, return_activations=False):
        # 1. Prepare Data
        geno_x, meta_x, parents_genos, child_genos, p_trait, c_trait = self._parse_data(data)
        kl_scale = 0.1 

        with tf.GradientTape(persistent=True) as tape:
            # --- Forward Pass A: Parents (P_Encoder) ---
            p_mean, p_logvar, _ = self.p_encoder(parents_genos, meta_x, training=True)
            p_embed_x = self.sample_z(tf.clip_by_value(p_mean, -30, 30), tf.clip_by_value(p_logvar, -7, 7))
            
            # Reconstruct parents' offspring (Synthetic)
            p_geno_logits, _, _, _ = self.decoder(parents_genos, p_embed_x, meta_x, training=True)
            p_pheno_pred, _, _ = self.regressor(p_trait, p_embed_x, parents_genos, training=True)

            # --- Forward Pass B: Standard (Encoder) ---
            mean, logvar, enc_act = self.encoder(geno_x, meta_x, training=True)
            embed_x = self.sample_z(tf.clip_by_value(mean, -30, 30), tf.clip_by_value(logvar, -7, 7))
            
            # Reconstruct actual offspring
            geno_logits, dec_act, dec_gate, _ = self.decoder(parents_genos, embed_x, meta_x, training=True)
            pheno_pred, reg_act, reg_gate = self.regressor(p_trait, embed_x, parents_genos, training=True)

            # --- Forward Pass C: Cyclic (New Genotype) ---
            # Create synthetic input from Parent Pass prediction and re-encode it
            _, new_embed_x, new_mean, new_logvar = self._generate_cyclic_offspring(
                p_geno_logits, parents_genos, meta_x, training=True
            )
            
            # Predict from cyclic embedding
            new_geno_logits, _, _, _ = self.decoder(parents_genos, new_embed_x, meta_x, training=True)
            new_pheno_pred, _, _ = self.regressor(p_trait, new_embed_x, parents_genos, training=True)

            # --- Loss Calculation ---
            # Real Data Loss
            kl_loss, _, reg_loss, rec_loss = self.loss_fn(
                child_genos, geno_logits, mean, logvar, 
                trait_pred=pheno_pred, trait_true=c_trait, epoch=cur_epoch, 
                p_mean=p_mean, p_logvar=p_logvar, training=True
            )
            # Synthetic/Parent Data Loss
            _, p_kl_loss, p_reg_loss, p_rec_loss = self.loss_fn(
                tf.stop_gradient(tf.argmax(p_geno_logits, -1)), # Target is what we just predicted
                new_geno_logits, mean, logvar, 
                trait_pred=new_pheno_pred, trait_true=p_pheno_pred, epoch=cur_epoch, 
                p_mean=p_mean, p_logvar=p_logvar, training=True
            )

            # Cyclic Consistency Loss (KL between Forward and Reverse)
            kl_forward = kl_divergence(p_mean, p_logvar, new_mean, new_logvar)
            kl_reverse = kl_divergence(new_mean, new_logvar, p_mean, p_logvar)
            cyclic_loss = 0.5 * (kl_forward + kl_reverse)

            # Aggregated Losses
            total_loss_kl = kl_loss * kl_scale + reg_loss + rec_loss * 1000
            total_p_loss = p_kl_loss * kl_scale + cyclic_loss * kl_scale + p_reg_loss + p_rec_loss * 1000

        # 3. Apply Gradients (Grouped by optimizer for clarity)
        pairs = [
            (self.opts['p_enc'], total_p_loss, self.p_encoder),
            (self.opts['enc'], total_loss_kl, self.encoder),
            (self.opts['dec'], rec_loss, self.decoder),
            (self.opts['reg'], reg_loss, self.regressor)
        ]
        for opt, loss, model in pairs:
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
        
        del tape

        # 4. Update Metrics
        results = {
            "rec_loss": rec_loss, "p_rec_loss": p_rec_loss, 
            "reg_loss": reg_loss, "p_reg_loss": p_reg_loss,
            "kl_loss": kl_loss, "p_kl_loss": p_kl_loss, 
            "kl_scale": kl_scale, "epoch": cur_epoch,
            "pop_shared_genos": self.pop_shared_genos(geno_logits, meta_x[2]),
            "trait_diff": tf.math.reduce_variance(pheno_pred)
        }
        self._update_metrics(results, c_trait, pheno_pred, child_genos, geno_logits, p_geno_logits, meta_x)
        all_activations = {"encoder": enc_act, "decoder": dec_act, "regressor": reg_act}
        return {**all_activations, **{m.name: m.result() for m in self.metrics}}
    
    @tf.function
    def test_step(self, data, cur_epoch=0):
            # 1. Prepare Data (Reuse the helper)
            geno_x, meta_x, parents_genos, child_genos, p_trait, c_trait = self._parse_data(data)

            # We generally don't scale KL during testing, or set it to 1.0 (full regularization)
            # depending on your preference.
            kl_scale = 1.0 

            # --- Forward Pass A: Parents (P_Encoder) ---
            p_mean, p_logvar, _ = self.p_encoder(parents_genos, meta_x, training=False)
            p_embed_x = self.sample_z(p_mean, p_logvar) # You might use mean directly for deterministic testing

            # Reconstruct parents' offspring (Synthetic)
            p_geno_logits, _, _, _ = self.decoder(parents_genos, p_embed_x, meta_x, training=False)
            p_pheno_pred, _, _ = self.regressor(p_trait, p_embed_x, parents_genos, training=False)

            # --- Forward Pass B: Standard (Encoder) ---
            mean, logvar, _ = self.encoder(geno_x, meta_x, training=False)
            embed_x = self.sample_z(mean, logvar)

            # Reconstruct actual offspring
            geno_logits, dec_act, dec_gate, _ = self.decoder(parents_genos, embed_x, meta_x, training=False)
            pheno_pred, reg_act, reg_gate = self.regressor(p_trait, embed_x, parents_genos, training=False)

            # --- Forward Pass C: Cyclic (New Genotype) ---
            _, new_embed_x, new_mean, new_logvar = self._generate_cyclic_offspring(
                p_geno_logits, parents_genos, meta_x, training=False
            )

            new_geno_logits, _, _, _ = self.decoder(parents_genos, new_embed_x, meta_x, training=False)
            new_pheno_pred, _, _ = self.regressor(p_trait, new_embed_x, parents_genos, training=False)

            # --- Loss Calculation ---
            # Note: We pass cur_epoch=0 or a fixed value since validation shouldn't depend on annealing
            kl_loss, _, reg_loss, rec_loss = self.loss_fn(
                child_genos, geno_logits, mean, logvar, 
                trait_pred=pheno_pred, trait_true=c_trait, epoch=0, 
                p_mean=p_mean, p_logvar=p_logvar, training=False
            )

            _, p_kl_loss, p_reg_loss, p_rec_loss = self.loss_fn(
                tf.stop_gradient(tf.argmax(p_geno_logits, -1)),
                new_geno_logits, mean, logvar, 
                trait_pred=new_pheno_pred, trait_true=p_pheno_pred, epoch=cur_epoch, 
                p_mean=p_mean, p_logvar=p_logvar, training=False
            )

            # 4. Update Metrics
            results = {
                "rec_loss": rec_loss, "p_rec_loss": p_rec_loss, 
                "reg_loss": reg_loss, "p_reg_loss": p_reg_loss,
                "kl_loss": kl_loss, "p_kl_loss": p_kl_loss, 
                "kl_scale": kl_scale, 
                "pop_shared_genos": self.pop_shared_genos(geno_logits, meta_x[2]),
                "trait_diff": tf.math.reduce_variance(pheno_pred)
            }

            self._update_metrics(results, c_trait, pheno_pred, child_genos, geno_logits, p_geno_logits, meta_x)

            return {m.name: m.result() for m in self.metrics}

    def _update_metrics(self, loss_dict, trait_true, trait_pred, geno_true, geno_logits, p_geno_logits, meta_x):
        """Centralized metric update logic."""
        # Update simple mean trackers
        for name, val in loss_dict.items():
            if name in self.trackers:
                self.trackers[name].update_state(val)
        
        # Update specific trackers
        self.trackers['mean_deviation'].update_state(trait_true[:, 0], trait_pred[:, 0])
        self.trackers['sd_deviation'].update_state(trait_true[:, 1], trait_pred[:, 1])
        self.trackers['cat_acc'].update_state(geno_true, geno_logits)
        self.trackers['p_cat_acc'].update_state(geno_true, p_geno_logits) # Note: Is label correct for p_acc?

        # Class-wise accuracy
        pred_class = tf.nn.softmax(geno_logits, axis=-1)
        for i, tracker in enumerate(self.class_acc_trackers):
            mask = tf.equal(geno_true, i)
            # Only update if class exists in batch
            if tf.reduce_any(mask):
                tracker.update_state(tf.boolean_mask(geno_true, mask), tf.boolean_mask(pred_class, mask))
        pop_ids = meta_x[2] 
        # trait_true/pred assumed shape: (Batch, 2) -> [Mean, Std]
        # Column 0 = Mean, Column 1 = Std
        
        self.pop_trackers['pop_true_mean'].update_state(trait_true[:, 0], pop_ids)
        self.pop_trackers['pop_pred_mean'].update_state(trait_pred[:, 0], pop_ids)
        
        self.pop_trackers['pop_true_std'].update_state(trait_true[:, 1], pop_ids)
        self.pop_trackers['pop_pred_std'].update_state(trait_pred[:, 1], pop_ids)

    @property
    def metrics(self):
        return list(self.trackers.values()) + list(self.pop_trackers.values()) + self.class_acc_trackers

    @tf.function
    def call(self, data, training=False, return_activations=False):
        # Simplified call method...
        # geno_x, meta_x, parents_genos, _, p_trait, _ = self._parse_data(data)
        geno_x, p_trait, meta_x = data
        parents_genos = geno_x[:, :2, ...]
        
        # Standard Encoder Pass
        _, _, enc_act = self.encoder(geno_x, meta_x, training=training, return_activations=return_activations)
        
        # Parent Encoder Pass
        p_mean, p_logvar, _ = self.p_encoder(parents_genos, meta_x, training=training, return_activations=return_activations)
        embed_x = self.sample_z(p_mean, p_logvar)

        # Decoder/Regressor Pass
        geno_logits, dec_act, dec_gate, geno_pred = self.decoder(
            parents_genos, embed_x, meta_x, training=training, return_activations=return_activations
        )
        pheno_pred, reg_act, reg_gate = self.regressor(
            p_trait, embed_x, parents_genos, training=training, return_activations=return_activations
        )

        if self.pheno_only:
            return pheno_pred
            
        all_activations = {"encoder": enc_act, "decoder": dec_act, "regressor": reg_act}
        return embed_x, geno_logits, pheno_pred, p_mean, p_logvar, dec_gate, reg_gate, all_activations, geno_pred