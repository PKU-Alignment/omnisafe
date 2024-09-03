# Experiment Grid

In the context of RL experiments, it is imperative to assess the performance of various algorithms across multiple environments. However, the inherent influence of randomness necessitates repeated evaluations employing distinct random seeds. To tackle this challenge, introduces an `Experiment Grid`, facilitating simultaneous initiation of multiple experimental sets. Researchers are merely required to pre-configure the experimental parameters, subsequently executing multiple experiment sets in parallel via a single file. An exemplification of this process can be found in <a href="#expgrid">Figure 1</a>.

<img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/grid/grid.png?raw=true" id="expgrid">
<br>
<caption><p><b></b></p></caption>

**Figure 1:** OmniSafe ’s `Experiment Grid`. The left side of the figure displays the main unction of the run `experiment_grid.py` file, while the right side shows the status of the `Experiment Grid` execution. In this example, three distinct random seeds are selected for the `SafetyAntVelocity-v1` and `SafetyWalker2dVelocity-v1`, then the PPOLag and TRPO-Lag algorithms are executed.

<img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/grid/example_ant.png?raw=true" id="ant">
<br>
<div style="color:orange; border-bottom: 1px solid #d9d9d9; text-align: center; color: #999; padding: 2px;">
SafetyAntVelocity-v1
</div>
<img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08); text-align: center;" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/grid/example_walker2d.png?raw=true" id="walker2d">
<div style="color:orange; border-bottom: 1px solid #d9d9d9; text-align: center; color: #999; padding: 2px;">
SafetyWalker2dVelocity-v1
</div>
<br>

**Figure 2:** Analysis of the example experiment results. The blue lines are the results from PPOLag, while the orange ones are TRPO-Lag. The solid line in the figure represents the mean of multiple random seeds, while the shadow represents the standard deviation among 0, 5, and 10 random seeds.

The `run_experiment_grid.py` script executes experiments in parallel based on user-configured parameters and generates corresponding graphs of the experimental results. In the example presented in <a href="#expgrid">Figure 1</a>, we specified that the script should draw curves based on different environments and obtained the training curves of PPOLag and TRPO-Lag in `SafetyAntVelocity-v1` and `SafetyWalker2dVelocity-v1`, where seeds have been grouped.

Moreover, combined with `Statistics Tools`, the `Experiment Grid` is a powerful tool for parameter tuning. As illustrated in <a href="#compare">Figure 3</a>, we utilized the `Experiment Grid` to explore the impact of `batch_size` on the performance of PPOLag and TRPO-Lag in `SafetyWalker2dVelocity-v1` and `SafetyAntVelocity-v1`, then used `Statistics Tools` to analyze the experiment results. It is obvious that the `batch_size` has a significant influence on the performance of PPOLag in `SafetyWalker2dVelocity-v1`, and the optimal `batch_size` is 128. Obtaining this conclusion requires repeating the experiment multiple times, and the `Experiment Grid` significantly expedites the process.

<img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/grid/compare.png?raw=true" id="compare">
<br>

**Figure 3:** An example of how the `Experiment Grid` can be utilized for parameter tuning. In this particular example, we set the `batch_size` in the `algo_cfgs` to 64, 128, and 256. Then we ran multiple experiments using the `Experiment Grid`, and finally used `Statistics Tools` to analyze the impact of the `batch_size` on the performance of the algorithm. Note that different colors denote different `batch_size`. The results showed that the `batch_size` had a significant effect on the performance of the algorithm, and the optimal `batch_size` was found to be 128. The `Experiment Grid` enabled us to efficiently explore the effect of different parameter values on the algorithm's performance.
