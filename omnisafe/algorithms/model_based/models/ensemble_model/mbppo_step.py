    # pylint: disable-next=too-many-locals
    def mbppo_step(self, obs, act, idx=None, deterministic=False):
        # pylint: disable-next=line-too-long
        """use numpy input to predict single next state by randomly select one model result or select index model result."""
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        if idx is None:
            idx = self.model.elite_model_idxes
        else:
            idx = [idx]
        inputs = np.concatenate((obs, act), axis=-1)
        ensemble_model_means, ensemble_model_vars = self.model.predict(inputs)
        ensemble_model_means[:, :, self.state_start_dim :] += obs

        ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = (
                ensemble_model_means
                + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds
            )

        _, batch_size, _ = ensemble_model_means.shape
        model_idxes = np.random.choice(idx, size=batch_size)
        batch_idxes = np.arange(0, batch_size)

        samples = ensemble_samples[model_idxes, batch_idxes]
        if self.algo == 'MBPPOLag' and self.model.env_type == 'mujoco-velocity':
            rewards, cost, next_obs = (
                samples[:, 0],
                samples[:, 1],
                samples[:, self.state_start_dim :],
            )
            terminals = self._termination_fn(self.env_name, obs, act, next_obs)
        elif self.algo == 'MBPPOLag' and self.model.env_type == 'gym':
            next_obs = samples
            rewards = None
            cost = None
            terminals = None

        if return_single:
            next_obs = next_obs[0]
            if self.model.env_type == 'mujoco-velocity':
                rewards = rewards[0]
                cost = cost[0]

        return next_obs, rewards, cost, terminals