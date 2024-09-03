# Supported Algorithms

OmniSafe offers a highly modular framework that integrates an extensive collection of algorithms specifically designed for Safe Reinforcement Learning (SafeRL) in various domains. The `Adapter` module in OmniSafe allows for easily expanding different types of SafeRL algorithms.

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
    <th>Domains</th>
    <th>Types</th>
    <th>Algorithms Registry</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="5">On Policy</td>
    <td rowspan="2">Primal Dual</td>
    <td>TRPOLag; PPOLag; PDO; RCPO</td>
  </tr>
  <tr>
    <td>TRPOPID; CPPOPID</td>
  </tr>
  <tr>
    <td>Convex Optimization</td>
    <td><span style="font-weight:400;font-style:normal">CPO; PCPO; </span>FOCOPS; CUP</td>
  </tr>
  <tr>
    <td>Penalty Function</td>
    <td>IPO; P3O</td>
  </tr>
  <tr>
    <td>Primal</td>
    <td>OnCRPO</td>
  </tr>
  <tr>
    <td rowspan="3">Off Policy</td>
    <td rowspan="2">Primal-Dual</td>
    <td>DDPGLag; TD3Lag; SACLag</td>
  </tr>
  <tr>
    <td><span style="font-weight:400;font-style:normal">DDPGPID; TD3PID; SACPID</span></td>
  </tr>
    <td rowspan="1">Control Barrier Function</td>
    <td>DDPGCBF, SACRCBF, CRABS</td>
  </tr>
  <tr>
    <td rowspan="2">Model-based</td>
    <td>Online Plan</td>
    <td>SafeLOOP; CCEPETS; RCEPETS</td>
  </tr>
  <tr>
    <td><span style="font-weight:400;font-style:normal">Pessimistic Estimate</span></td>
    <td>CAPPETS</td>
  </tr>
    <td rowspan="2">Offline</td>
    <td>Q-Learning Based</td>
    <td>BCQLag; C-CRR</td>
  </tr>
  <tr>
    <td>DICE Based</td>
    <td>COptDICE</td>
  </tr>
  <tr>
    <td rowspan="3">Other Formulation MDP</td>
    <td>ET-MDP</td>
    <td><span style="font-weight:400;font-style:normal">PPO</span>EarlyTerminated; TRPOEarlyTerminated</td>
  </tr>
  <tr>
    <td>SauteRL</td>
    <td>PPOSaute; TRPOSaute</td>
  </tr>
  <tr>
    <td>SimmerRL</td>
    <td><span style="font-weight:400;font-style:normal">PPOSimmerPID; TRPOSimmerPID</span></td>
  </tr>
</tbody>
</table>
</div>

<caption><p><b>Table 1:</b> OmniSafe supports varieties of SafeRL algorithms. From the perspective of classic RL, OmniSafe includes on-policy, off-policy, offline, and model-based algorithms; From the perspective of the SafeRL learning paradigm, OmniSafe supports primal-dual, projection, penalty function, primal, etc.</p></caption>
