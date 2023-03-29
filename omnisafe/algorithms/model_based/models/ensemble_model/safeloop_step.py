    # pylint: disable-next=too-many-arguments,too-many-locals
    def safeloop_step(self, obs, act, deterministic=False, all_model=False, repeat_network=False):
        """Use tensor input to predict single next state by randomly select elite model result for online planning"""
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]

        inputs = torch.cat((obs, act), dim=-1)
        ensemble_model_means, ensemble_model_vars = self.model.predict_t(
            inputs, repeat_network=repeat_network
        )

        ensemble_model_means[:, :, self.state_start_dim :] += obs

        ensemble_model_stds = torch.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = (
                ensemble_model_means
                + torch.randn(size=ensemble_model_means.shape).to(self.device) * ensemble_model_stds
            )

        # use all dynamics model result
        if all_model:
            samples = ensemble_samples
            samples_var = ensemble_model_vars
        # only use elite model result
        else:
            _, batch_size, _ = ensemble_model_means.shape
            model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)
            batch_idxes = np.arange(0, batch_size)
            samples = ensemble_samples[model_idxes, batch_idxes]
            samples_var = ensemble_model_vars[model_idxes, batch_idxes]

        return samples, samples_var