# Case Study

One important motivation for SafeRL is to enable agents to explore and
learn safely. Therefore, evaluating algorithm performance concerning
*procedural constraint violations* is also important. We have selected
representative experimental results and report as shown in <a href="#analys">Figure 1</a> and <a href="#analys_ppo">Figure 2</a>:

#### Radical vs. Conservative

*Radical* policies often explore higher rewards but violate more safety
constraints, whereas *Conservative* policies are the opposite.
<a href="#analys">Figure 1</a> illustrates this: during training, CPO and
PPOLag consistently pursue the highest rewards among all algorithms, as
depicted in the first row. However, as shown in the second row, they
experience significant fluctuations in constraint violations, especially
for PPOLag. So, they are relatively radical, *i.e.,* higher rewards but
higher costs. In comparison, while P3O achieves slightly lower rewards
than PPOLag, it maintains fewer oscillations in constraint violations,
making it safer in adhering to safety constraints, evident from the
smaller proportion of its distribution crossing the black dashed line. A
similar pattern is also observed when comparing PCPO with CPO.
Therefore, P3O and PCPO are relatively conservative, *i.e.,* lower costs
but lower rewards.

<img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/analys.png?raw=true" id="analys">
<br>

**Figure 1:** PPOLag, P3O, CPO, and PCPO training on four tasks in for 1e7 steps, showing the distribution of all episodic rewards and costs. All data covers over 5 random seeds and filters out data points over 3 standard deviations. The black dashed line in the graph represents the preset `cost_limit`.


#### Oscillation vs. Stability

The oscillations in the degree of constraint violations during the
training process can indicate the performance of SafeRL algorithms.
These oscillations are quantified by *Extremes*, *i.e.,* the maximum
constraint violation, and *Distributions*, *i.e.,* the frequency of
violations remaining below a predefined `cost_limit`. As shown in
<a href="#analys_ppo">Figure 2</a>, PPOLag, a popular baseline in SafeRL,
utilizes the Lagrangian multiplier for constraint handling. Despite its
simplicity and ease of implementation, PPOLag often suffers from
significant oscillations due to challenges in setting appropriate
initial values and learning rates. It consistently seeks higher rewards
but always leads to larger extremes and unsafe distributions.
Conversely, CPPOPID, which employs a PID controller for updating the
Lagrangian multiplier, markedly reduces these extremes. CUP implements a
two-stage projection method that constrains violations' distribution
below the `cost_limit`. Lastly, PPOSaute integrates state observations
with constraints, resulting in smaller extremes and safer distributions
of violations.

<img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Gaiejj/omnisafe_benchmarks_cruve/blob/main/on-policy/benchmarks/analys_ppo.png?raw=true"  id="analys_ppo">
<br>

**Figure 2:** PPOLag, CPPOPID, CUP, and PPOSaute trained on four tasks in for all 1e7 steps, showing the distribution of all episodic rewards and costs. All data covers over 5 random seeds and filters out data points over 3 standard deviations. The black dashed line in the graph represents the preset `cost_limit`.
