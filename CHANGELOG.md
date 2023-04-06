# Changelog

<!-- markdownlint-disable no-duplicate-header -->

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Features

- Feat: add `ruff` and `codespell` integration by [@XuehaiPan](https://github.com/XuehaiPan) in PR [#186](https://github.com/OmniSafeAI/omnisafe/pull/186).

### Fixes

### Documentation



## v0.2.2

### Fixes
- Add MANIFEST.in by [@Borong Zhang](https://github.com/muchvo) in PR [#182](https://github.com/OmniSafeAI/omnisafe/pull/182).

### Documentation
- Update api documentation by [@Jiayi Zhou](https://github.com/Gaiejj) in PR [#181](https://github.com/OmniSafeAI/omnisafe/pull/181)

## v0.2.1

### Features
- Feat(statistics tools): support statistics tools for experiments by [@Borong Zhang](https://github.com/muchvo) in PR [#157](https://github.com/OmniSafeAI/omnisafe/pull/157).

## v0.2.0

### Features

- Support cuda by [@Jiayi Zhou](https://github.com/Gaiejj) in PR [#163](https://github.com/OmniSafeAI/omnisafe/pull/163).
- Support command line interfaces for omnisafe by [@Borong Zhang](https://github.com/muchvo) in PR [#144](https://github.com/OmniSafeAI/omnisafe/pull/144).
- Refactor(wrapper): refactor the cuda setting by [@Jiayi Zhou](https://github.com/Gaiejj) in PR [#176](https://github.com/OmniSafeAI/omnisafe/pull/176).

### Fixes

- Fix(onpolicy_adapter): fix the calculation of last state value by [@Borong Zhang](https://github.com/muchvo) in PR [#164](https://github.com/OmniSafeAI/omnisafe/pull/164).
- Fix(config.py): fix config assertion by [@Jiayi Zhou](https://github.com/Gaiejj) in PR [#174](https://github.com/OmniSafeAI/omnisafe/pull/174).
- Fix autoreset wrapper in by [@r-y1](https://github.com/r-y1) PR [#167](https://github.com/OmniSafeAI/omnisafe/pull/167).

### Documentation

- Update docs style by [@Jiayi Zhou](https://github.com/Gaiejj) in PR [#169](https://github.com/OmniSafeAI/omnisafe/pull/169).
- Fix typo in readme by [@Ruiyang Sun](https://github.com/rockmagma02) in PR [#172](https://github.com/OmniSafeAI/omnisafe/pull/172).
- Update README and the usage of CLI by [@Jiaming Ji](https://github.com/zmsn-2077) in PR [#138](https://github.com/OmniSafeAI/omnisafe/pull/138).
## v0.1.0
- Check out `Development` for more.
------
## Development
### 2023-03-06 ~ 2023-03-15
#### Features

- Chore(on-policy): update benchmark performance for first-order algorithms by [@Borong Zhang](https://github.com/muchvo) in PR [#148](https://github.com/OmniSafeAI/omnisafe/pull/148).
- Feat(off-policy): add DDPG, TD3 SAC by [@Jiayi Zhou](https://github.com/Gaiejj) in PR [#128](https://github.com/OmniSafeAI/omnisafe/pull/128).
- Feat: support policy evaluation by [@Jiayi Zhou](https://github.com/Gaiejj) in PR [#137](https://github.com/OmniSafeAI/omnisafe/pull/137).
- Test: add more test case, and fix bugs by [@Ruiyang Sun](https://github.com/rockmagma02) in PR [#136](https://github.com/OmniSafeAI/omnisafe/pull/136).
- Fix(logger, wrapper): support csv file and velocity tasks by [@Jiayi Zhou](https://github.com/Gaiejj) in PR [#131](https://github.com/OmniSafeAI/omnisafe/pull/131).
- Feat: update architecture of config.yaml by [@Jiaming Ji](https://github.com/zmsn-2077) in PR [#126](https://github.com/OmniSafeAI/omnisafe/pull/126).
- Chore: support num_thread setting by [@Jiayi Zhou](https://github.com/Gaiejj) in PR [#124](https://github.com/OmniSafeAI/omnisafe/pull/124).
- Refactor: change architecture of omnisafe by [@Ruiyang Sun](https://github.com/rockmagma02) in PR [#121](https://github.com/OmniSafeAI/omnisafe/pull/121).

#### Fixes

- Fix(on-policy): fix the second order algorithms performance by [@Jiayi Zhou](https://github.com/Gaiejj)  in PR [#147](https://github.com/OmniSafeAI/omnisafe/pull/147).
- Fix(rollout, exp_grid): fix logdir path conflict by [@Borong Zhang](https://github.com/muchvo) in PR [#145](https://github.com/OmniSafeAI/omnisafe/pull/145).
- Fix: support new config for exp_grid by [@Borong Zhang](https://github.com/muchvo) in PR [#142](https://github.com/OmniSafeAI/omnisafe/pull/142).
- Fix(ppo): fix entropy loss by [@Jiayi Zhou](https://github.com/Gaiejj) in PR [#135](https://github.com/OmniSafeAI/omnisafe/pull/135).
- Fix(algo): fix no return in algo_wrapper::learn by [@Ruiyang Sun](https://github.com/rockmagma02) in PR [#122](https://github.com/OmniSafeAI/omnisafe/pull/122).

#### Documentation
- Docs: Update changelog by [@Jiaming Ji](https://github.com/zmsn-2077).
- Docs: Update README.md: fix action passing by [@Jiayi Zhou](https://github.com/Gaiejj) in PR [#149](https://github.com/OmniSafeAI/omnisafe/pull/149).
- Chore: fix typo by [@1Asan](https://github.com/1Asan) in PR [#134](https://github.com/OmniSafeAI/omnisafe/pull/134).

### 2023-02-27 ~ 2023-03-05
#### Fixes

- Fix(P3O): fix P3O performance by [@Jiayi Zhou](https://github.com/Gaiejj) in PR [#123](https://github.com/OmniSafeAI/omnisafe/pull/123).
- Fix(off-policy): fix `action passing` by [@Jiayi Zhou](https://github.com/Gaiejj) in PR [#119](https://github.com/OmniSafeAI/omnisafe/pull/119).

#### Documentation
- Docs: update logo by [@Jiayi Zhou](https://github.com/Gaiejj) in PR [#125](https://github.com/OmniSafeAI/omnisafe/pull/125).

### 2023-02-13 ~ 2023-02-19

#### Fixes

- Fix(evaluator): fix evaluator by [@Ruiyang Sun](https://github.com/rockmagma02) in PR [#117](https://github.com/OmniSafeAI/omnisafe/pull/117).

### 2023-02-06 ~ 2023-02-12

#### Features

- Build(env): delete local `safety-gymnaisum` dependence by [@Ruiyang Sun](https://github.com/rockmagma02) in PR [#102](https://github.com/OmniSafeAI/omnisafe/pull/102).
- Refactor(buffer): refactor `buffer` by [@Ruiyang Sun](https://github.com/rockmagma02) in PR [#101](https://github.com/OmniSafeAI/omnisafe/pull/101).
- Refactor: clean the code by [@Jiayi Zhou](https://github.com/Gaiejj) in
#### Fixes

- Fix: fix tools by [@Jiayi Zhou](https://github.com/Gaiejj) in PR [#100](https://github.com/OmniSafeAI/omnisafe/pull/100).
- Fix: fix algo wrapper by [@Jiayi Zhou](https://github.com/Gaiejj) in PR [#99](https://github.com/OmniSafeAI/omnisafe/pull/99).
PR [#97](https://github.com/OmniSafeAI/omnisafe/pull/97).
#### Documentation

- Modify `logo.png` and add `requirements.txt` by [@Ruiyang Sun](https://github.com/rockmagma02) in PR [#103](https://github.com/OmniSafeAI/omnisafe/pull/103).


### 2023-01-30 ~ 2023-02-05

#### Features

- Chore: update linter settings by [@XuehaiPan](https://github.com/XuehaiPan).
- Chore: update ci  by [@Jiayi Zhou](https://github.com/Gaiejj) in PR [#90](https://github.com/OmniSafeAI/omnisafe/pull/90) reviewed by [@Jiaming Ji](https://github.com/zmsn-2077) and [@friedmainfunction](https://github.com/friedmainfunction).
- Chore: update yaml  by [@Jiayi Zhou](https://github.com/Gaiejj) in PR [#92](https://github.com/OmniSafeAI/omnisafe/pull/92) and [#93](https://github.com/OmniSafeAI/omnisafe/pull/93) reviewed by [@Jiaming Ji](https://github.com/zmsn-2077) and [@friedmainfunction](https://github.com/friedmainfunction).

### 2023-01-23 ~ 2023-01-29

#### Features

- Refactor(objects): change object type into free_geom by [@Borong Zhang](https://github.com/muchvo) in PR [#89](https://github.com/OmniSafeAI/omnisafe/pull/89).
- Chore: update algorithms configuration by [@Jiayi Zhou](https://github.com/Gaiejj) in PR [#88](https://github.com/OmniSafeAI/omnisafe/pull/88).
- Feat: support cuda by [@Jiayi Zhou](https://github.com/Gaiejj in PR [#86](https://github.com/OmniSafeAI/omnisafe/pull/86).
- Feat(render): add keyboard debug mode for some agents in all tasks by [@Borong Zhang](https://github.com/muchvo) in PR [#83](https://github.com/OmniSafeAI/omnisafe/pull/83).
- Feat: add experiment grid by [@Jiaming Ji](https://github.com/zmsn-2077) in PR [#84](https://github.com/OmniSafeAI/omnisafe/pull/84).

#### Fixes

- Fix seed setting  by [@Jiayi Zhou](https://github.com/Gaiejj) in PR [#82](https://github.com/OmniSafeAI/omnisafe/pull/82).


### 2023-01-16 ~ 2023-01-22

#### Features

- Feat(agents): add `ant` agent by [@Borong Zhang](https://github.com/muchvo) in PR [#82](https://github.com/OmniSafeAI/omnisafe/pull/82).
- Refactor(safety-gymnaisum): `code decoupling` by [@Borong Zhang](https://github.com/muchvo) in PR [#81](https://github.com/OmniSafeAI/omnisafe/pull/81).
- Feat: add new algorithm by [@Jiayi Zhou](https://github.com/Gaiejj) in PR [#80](https://github.com/OmniSafeAI/omnisafe/pull/80).

### 2023-01-09 ~ 2023-01-15

#### Features

- Refactor: change wrapper setting by [@Jiayi Zhou](https://github.com/Gaiejj) in PR [#73](https://github.com/OmniSafeAI/omnisafe/pull/73).
- Feat: `vectorized` environment by [@Jiayi Zhou](https://github.com/Gaiejj) in PR [#74](https://github.com/OmniSafeAI/omnisafe/pull/74).
### 2023-01-02 ~ 2023-01-08

#### Features

- Feat(agents, tasks, Evaluator): support `circle012` and new agent `racecar`, update evaluator by [@Borong Zhang](https://github.com/muchvo) in PR [#59](https://github.com/OmniSafeAI/omnisafe/pull/59).

### 2022-12-26 ~ 2023-01-01

#### Features

- Refactor: enhanced model-based code, add `CAP` algorithm by [@Weidong Huang](https://github.com/hdadong) in PR [#59](https://github.com/OmniSafeAI/omnisafe/pull/59).
- Feat: support auto render as .mp4 videos, add examples and tests by [@Borong Zhang](https://github.com/muchvo) in PR [#60](https://github.com/OmniSafeAI/omnisafe/pull/60).
#### Fixes

- Fix(model-based): fix cap cost bug and lag beta value in cap.yaml by [@Weidong Huang](https://github.com/hdadong) in PR [#62](https://github.com/OmniSafeAI/omnisafe/pull/62).
- Fix(render): fix markers are not shown in the rgb array returned by env.render() by [@Borong Zhang](https://github.com/muchvo) in PR [#61](https://github.com/OmniSafeAI/omnisafe/pull/61).



### 2022-12-19 ~ 2022-12-25

#### Features

- Feat(circle, run): support new tasks by [@Borong Zhang](https://github.com/muchvo) in PR [#50](https://github.com/OmniSafeAI/omnisafe/pull/50).
- Add Makefile by [@XuehaiPan](https://github.com/XuehaiPan) in PR [#53](https://github.com/OmniSafeAI/omnisafe/pull/53).
#### Fixes

- Fix bug for namedtuple by [@Jiayi Zhou](https://github.com/Gaiejj) in PR [#54](https://github.com/OmniSafeAI/omnisafe/pull/54).
#### Documentation

- Fix spelling error by [@Jiayi Zhou](https://github.com/Gaiejj) in PR [#56](https://github.com/OmniSafeAI/omnisafe/pull/56), reviewed by [@Jiaming Ji](https://github.com/zmsn-2077) and [@XuehaiPan](https://github.com/XuehaiPan).


### 2022-12-12 ~ 2022-12-18

#### Features

- Refactor: open pylint in pre-commit by [@Jiaming Ji](https://github.com/zmsn-2077) in PR [#48](https://github.com/OmniSafeAI/omnisafe/pull/48).
- Refactor: change the details and yaml files of on policy algorithm by [@Jiayi Zhou](https://github.com/Gaiejj) in PR [#41](https://github.com/OmniSafeAI/omnisafe/pull/41).
- Feat: add CUP algorithm by [@Jiayi Zhou](https://github.com/Gaiejj) in PR [#43](https://github.com/OmniSafeAI/omnisafe/pull/43).
- Feat(wrapper): separated wrapper for different algorithmic environments by [@Jiaming Ji](https://github.com/zmsn-2077) in PR [#44](https://github.com/OmniSafeAI/omnisafe/pull/44).
- Chore: rename files and enable pylint by [@Borong Zhang](https://github.com/muchvo) in  PR [#39](https://github.com/OmniSafeAI/omnisafe/pull/39).
#### Documentation

- Retouch the formatting and add PPO docs for omnisafe by [@Jiayi Zhou](https://github.com/Gaiejj) in PR [#40](https://github.com/OmniSafeAI/omnisafe/pull/40).
- Add Lagrangian method documentation by [@Jiayi Zhou](https://github.com/Gaiejj) in PR [#42](https://github.com/OmniSafeAI/omnisafe/pull/42).
- Refactor(README): show the implemented algorithms in more detail by [@Jiaming Ji](https://github.com/zmsn-2077) in PR [#47](https://github.com/OmniSafeAI/omnisafe/pull/47).
### 2022-12-05 ~ 2022-12-11

#### Features

- Refactor: more OOP style code were used and made better code and file structure by [@Borong Zhang](https://github.com/muchvo) in PR [#37](https://github.com/OmniSafeAI/omnisafe/pull/37).
- Refactor: change the file layout of omnisafe by [@Jiaming Ji](https://github.com/zmsn-2077) in PR [#35](https://github.com/OmniSafeAI/omnisafe/pull/35).
#### Fixes

- Fix(env_wrapper): fix warning caused by 'none' string default value by [@Borong Zhang](https://github.com/muchvo) in PR [#30](https://github.com/OmniSafeAI/omnisafe/pull/30).
#### Documentation

- Docs: retouch the formatting and add links to the formula numbers by [@Jiayi Zhou](https://github.com/Gaiejj) in PR [#31](https://github.com/OmniSafeAI/omnisafe/pull/31).
### 2022-11-28 ~ 2022-12-04

#### Features

- Chore(.github): update issue templates by [@XuehaiPan](https://github.com/XuehaiPan) in PR [#29](https://github.com/OmniSafeAI/omnisafe/pull/29).
- Refactor packaging by [@XuehaiPan](https://github.com/XuehaiPan) in PR [#20](https://github.com/OmniSafeAI/omnisafe/pull/20).
- Add ddpg, clean some code, modify algo_wrapper in PR [#24](https://github.com/OmniSafeAI/omnisafe/pull/24) by [@Jiaming Ji](https://github.com/zmsn-2077).
#### Documentation

- Add `TRPO` to docs by [@Jiayi Zhou](https://github.com/Gaiejj) in PR [#28](https://github.com/OmniSafeAI/omnisafe/pull/28).
- Add `FOCOPS` and `PCPO` to docs by [@XuehaiPan](https://github.com/XuehaiPan) in [#21](https://github.com/OmniSafeAI/omnisafe/pull/21).

### 2022-11-20 ~ 2022-11-27

#### Features

- Add render_mode: `human`, `rgb_array`, `depth_array` in safety-gymnasium: `safety_gym_v2`.
- Add **Model-based Safe Algorithms:** `mbppolag`, `safeloop` by [@Weidong Huang](https://github.com/hdadong) in [#12](https://github.com/OmniSafeAI/omnisafe/pull/12).
- Add .editorconfig and update license by [@XuehaiPan](https://github.com/XuehaiPan) in [#8](https://github.com/OmniSafeAI/omnisafe/pull/8).
#### Fixes

- Fix readme typo by [@erjanmx](https://github.com/erjanmx) in PR [#13](https://github.com/OmniSafeAI/omnisafe/pull/13).
- Fix ambiguous config yaml for algorithms by [@Jiaming Ji](https://github.com/zmsn-2077) in PR [#6](https://github.com/OmniSafeAI/omnisafe/pull/6).
- Fix vis `safety_gym_v2` with del the render_mode by [@Jiaming Ji](https://github.com/zmsn-2077) in PR [#3](https://github.com/OmniSafeAI/omnisafe/pull/3).

#### Documentation

- Refactor some code in omnisafe, add `CHANGELOG.md`,` and del install.md and tutorial in PR [#16](https://github.com/OmniSafeAI/omnisafe/pull/16) by [@Jiaming Ji](https://github.com/zmsn-2077).
- Docs: add `PCPO` in omnisafe's **docs** and modify `CPO` by [@Jiayi Zhou](https://github.com/Gaiejj) in [#9](https://github.com/OmniSafeAI/omnisafe/pull/9).
- Add `CPO` and `Intro` in omnisafe's **docs** by [@Jiayi Zhou](https://github.com/Gaiejj) in PR [#7](https://github.com/OmniSafeAI/omnisafe/pull/7).
- Add render mode and vision input in safety-gymnasium: `safety_gym_v2` by [@Jiaming Ji](https://github.com/zmsn-2077) in PR [#5](https://github.com/OmniSafeAI/omnisafe/pull/5).
