# Features

OmniSafe transcends its role as a mere SafeRL library, functioning concurrently as a standardized and user-friendly SafeRL infrastructure. We compared the features of OmniSafe with popular open-source RL libraries. [See comparison results](#compare_with_repo).

> **Note:** All results in [compare_with_repo](#compare_with_repo) are accurate as of 2024. Please consider the latest results if you find any discrepancies between these data.

**Table 1:** Comparison of OmniSafe to a representative subset of RL or SafeRL libraries.

<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  .scrollable-container {
    overflow-x: auto;
    white-space: nowrap;
    width: 100%;
  }
  table {
    border-collapse: collapse;
    width: auto;
    font-size: 12px;
  }
  th, td {
    padding: 8px;
    text-align: center;
    border: 1px solid #ddd;
  }
  th {
    font-weight: bold;
  }
  caption {
    font-size: 12px;
    font-family: 'Times New Roman', Times, serif;
  }
</style>
</head>
<body>

<div class="scrollable-container">
<table>
    <thead>
        <tr>
            <th class="feature">Features</th>
            <th>OmniSafe</th>
            <th>TianShou</th>
            <th>Stable-Baselines3</th>
            <th>SafePO</th>
            <th>RL-Safety-Algorithms</th>
            <th>Safety-starter-agents</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td class="feature">Algorithm Tutorial</td>
            <td>✓</td>
            <td>✓</td>
            <td>✓</td>
            <td>✗</td>
            <td>✗</td>
            <td>✗</td>
        </tr>
        <tr>
            <td class="feature">API Documentation</td>
            <td>✓</td>
            <td>✓</td>
            <td>✓</td>
            <td>✓</td>
            <td>✗</td>
            <td>✗</td>
        </tr>
        <tr>
            <td class="feature">Command Line Interface</td>
            <td>✓</td>
            <td>✓</td>
            <td>✓</td>
            <td>✗</td>
            <td>✗</td>
            <td>✗</td>
        </tr>
        <tr>
            <td class="feature">Custom Environment</td>
            <td>✓</td>
            <td>✓</td>
            <td>✓</td>
            <td>✗</td>
            <td>✗</td>
            <td>✗</td>
        </tr>
        <tr>
            <td class="feature">Docker Support</td>
            <td>✓</td>
            <td>✓</td>
            <td>✓</td>
            <td>✗</td>
            <td>✗</td>
            <td>✗</td>
        </tr>
        <tr>
            <td class="feature">GPU Support</td>
            <td>✓</td>
            <td>✓</td>
            <td>✓</td>
            <td>✓</td>
            <td>✗</td>
            <td>✗</td>
        </tr>
        <tr>
            <td class="feature">Ipython / Notebook</td>
            <td>✓</td>
            <td>✓</td>
            <td>✓</td>
            <td>✗</td>
            <td>✗</td>
            <td>✗</td>
        </tr>
        <tr>
            <td class="feature">PEP8 Code Style</td>
            <td>✓</td>
            <td>✓</td>
            <td>✓</td>
            <td>✓</td>
            <td>✓</td>
            <td>✓</td>
        </tr>
        <tr>
            <td class="feature">Statistics Tools</td>
            <td>✓</td>
            <td>✓</td>
            <td>✓</td>
            <td>✗</td>
            <td>✗</td>
            <td>✗</td>
        </tr>
        <tr>
            <td class="feature">Test Coverage</td>
            <td>97%</td>
            <td>91%</td>
            <td>96%</td>
            <td>91%</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td class="feature">Type Hints</td>
            <td>✓</td>
            <td>✓</td>
            <td>✓</td>
            <td>✓</td>
            <td>✓</td>
            <td>✓</td>
        </tr>
        <tr>
            <td class="feature">Vectorized Environments</td>
            <td>✓</td>
            <td>✓</td>
            <td>✓</td>
            <td>✓</td>
            <td>✗</td>
            <td>✗</td>
        </tr>
        <tr>
            <td class="feature">Video Examples</td>
            <td>✓</td>
            <td>✓</td>
            <td>✓</td>
            <td>✗</td>
            <td>✗</td>
            <td>✗</td>
        </tr>
    </tbody>
</table>
</div>

<a id="compare_with_repo"></a> Compared to classic RL open-source libraries, [TianShou](https://www.jmlr.org/papers/v23/21-1127.html) and [Stable-Baselines3](https://jmlr.org/papers/v22/20-1364.html), OmniSafe adheres to the same engineering standards and supports user-friendly features. Compared to the SafeRL library, [SafePO](https://proceedings.neurips.cc/paper_files/paper/2023/file/3c557a3d6a48cc99444f85e924c66753-Paper-Datasets_and_Benchmarks.pdf), [RL-Safety-Algorithms](https://github.com/SvenGronauer/RL-Safety-Algorithms), and [Safety-starter-agents](https://github.com/openai/safety-starter-agents), OmniSafe offers greater ease of use and robustness, making it a foundational infrastructure to accelerate SafeRL research. The complete codebase of OmniSafe adheres to the PEP8 style, with each commit undergoing stringent evaluations, such as `isort`, `pylint`, `black`, and `ruff`. Before merging into the main branch, code modifications necessitate approval from at least two reviewers. These features enhance the reliability of OmniSafe and provide assurances for effective ongoing development.

OmniSafe includes a tutorial on `Colab` that provides a step-by-step guide to the training process, as illustrated in [Figure 2](#figure_2). For those who are new to SafeRL, the tutorial allows for interactive learning of the training procedure. By clicking on `Colab Tutorial`, users can access it and follow the instructions to understand better how to use OmniSafe. Seasoned researchers can capitalize on OmniSafe's informative command-line interface, as demonstrated in [Figure 1](#figure_1) and [Figure 3](#figure_3), facilitating rapid comprehension of the platform's utilization to expedite their scientific investigations.

Regarding the experiment execution process, OmniSafe presents an array of tools for analyzing experimental outcomes, encompassing `WandB`, `TensorBoard`, and `Statistics Tools`. Furthermore, OmniSafe has submitted its experimental benchmark to the `WandB` report [1], as depicted in [Figure 4](#figure_4). This report furnishes more detailed training curves and evaluation demonstrations of classic algorithms, serving as a valuable reference for researchers.

[1]: [https://api.wandb.ai/links/pku_rl/mv1eeetb](https://api.wandb.ai/links/pku_rl/mv1eeetb) | [https://api.wandb.ai/links/pku_rl/scvni0oj](https://api.wandb.ai/links/pku_rl/scvni0oj)

[cli]: #cli
[tutorial]: #tutorial
[cli_details]: #cli_details
[wandb_video]: #wandb_video


<img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/muchvo/omnisafe_docs_img/blob/main/features/cli_help-1.png?raw=true" id="analys">
<br>

<a id="figure_1"></a> **Figure 1:** An illustration of the OmniSafe command line interface. Users can view the commands supported by OmniSafe and a brief usage guide by simply typing `omnisafe --help` in the command line. If a user wants to further understand how to use a specific command, they can obtain additional prompts by using the command `omnisafe COMMAND --help`, as shown in [Figure 3](#figure_3).

<img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/muchvo/omnisafe_docs_img/blob/main/features/tutorial-1.png?raw=true" id="analys">
<br>

<a id="figure_2"></a> **Figure 2:** A example demonstrating the Colab tutorial provided by OmniSafe for using the `Experiment Grid`. The tutorial includes detailed usage descriptions and allows users to try running it and then see the results.

<img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/muchvo/omnisafe_docs_img/blob/main/features/cli_analyze_grid_help-1.png?raw=true" id="analys">
<br>

(a) Example of `omnisafe analyze-grid --help` in command line.

<img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/muchvo/omnisafe_docs_img/blob/main/features/cli_benchmark_help-1.png?raw=true" id="analys">
<br>

(b) Example of `omnisafe benchmark --help` in command line.

<img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/muchvo/omnisafe_docs_img/blob/main/features/cli_eval_help-1.png?raw=true" id="analys">
<br>

(c) Example of `omnisafe eval --help` in command line.

<img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/muchvo/omnisafe_docs_img/blob/main/features/cli_train_cfgs_help-1.png?raw=true" id="analys">
<br>

(d) Example of `omnisafe train-config --help` in command line.

<img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/muchvo/omnisafe_docs_img/blob/main/features/cli_help-1.png?raw=true" id="analys">
<br>

<a id="figure_3"></a> **Figure 3:** Here are some more details on using `omnisafe --help` command. Users can input `omnisafe COMMAND --help` to get help, where `COMMAND` includes all the items listed in `Commands` of [Figure 1](#figure_1). This feature enables users to swiftly acquire proficiency in executing common operations provided by OmniSafe via command-line and customize them further to meet their specific requirements.

<table style="width: 100%; border-collapse: collapse;">
    <tr>
        <td style="text-align: center; padding: 10px;">
            <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12), 0 2px 10px 0 rgba(34,36,38,.08); width: 100%;" src="https://github.com/muchvo/omnisafe_docs_img/blob/main/features/wandb_pointgoal-1.png?raw=true" alt="SafetyPointGoal1-v0" />
            <br>
            <strong>(a) SafetyPointGoal1-v0</strong>
        </td>
        <td style="text-align: center; padding: 10px;">
            <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12), 0 2px 10px 0 rgba(34,36,38,.08); width: 100%;" src="https://github.com/muchvo/omnisafe_docs_img/blob/main/features/wandb_pointbutton-1.png?raw=true" alt="SafetyPointButton1-v0" />
            <br>
            <strong>(b) SafetyPointButton1-v0</strong>
        </td>
    </tr>
    <tr>
        <td style="text-align: center; padding: 10px;">
            <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12), 0 2px 10px 0 rgba(34,36,38,.08); width: 100%;" src="https://github.com/muchvo/omnisafe_docs_img/blob/main/features/wandb_cargoal-1.png?raw=true" alt="SafetyCarGoal1-v0" />
            <br>
            <strong>(c) SafetyCarGoal1-v0</strong>
        </td>
        <td style="text-align: center; padding: 10px;">
            <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12), 0 2px 10px 0 rgba(34,36,38,.08); width: 100%;" src="https://github.com/muchvo/omnisafe_docs_img/blob/main/features/wandb_carbutton-1.png?raw=true" alt="SafetyCarButton1-v0" />
            <br>
            <strong>(d) SafetyCarButton1-v0</strong>
        </td>
    </tr>
</table>

<a id="figure_4"></a> <p><strong>Figure 4:</strong> An exemplification of OmniSafe's <code>WandB</code> reports videos. This example supplies videos of PPO and PPOLag in <code>SafetyPointGoal1-v0</code>, <code>SafetyPointButton1-v0</code>, <code>SafetyCarGoal1-v0</code>, and <code>SafetyCarButton1-v0</code> environments. The left of each sub-figure is PPO, while the right is PPOLag. Through these videos, we can intuitively witness the difference between safe and unsafe behavior. This is exactly what OmniSafe pursues: not just the safety of the training curve, but the true safety in a real sense.</p>


<img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/muchvo/omnisafe_docs_img/blob/main/features/wandb_curve-1.png?raw=true" id="analys">
<br>

**Figure 5:** An exemplification of OmniSafe's `WandB` reports training curve in `SafetyPointGoal1-v0`: The left panel represents the episode reward, and the right panel denotes the episode cost, with both encompassing the performance over 1e7 steps.
