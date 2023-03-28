# Changelog

<!-- markdownlint-disable no-duplicate-header -->

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v0.1.0
- Check out `Development` for more.

------
## Development
### 2023-03-06 ~ 2023-03-15
- Docs: update changelog by [@zmsn-2077](https://github.com/zmsn-2077).
- Docs: update README.md: fix action passing by [@Gaiejj](https://github.com/Gaiejj) in PR [#149](https://github.com/PKU-MARL/omnisafe/pull/149) reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@friedmainfunction](https://github.com/friedmainfunction).
- Chore(on-policy): update benchmark performance for first-order algorithms by [@muchvo](https://github.com/muchvo) in PR [#148](https://github.com/PKU-MARL/omnisafe/pull/148) reviewed by [@zmsn-2077](https://github.com/zmsn-2077).
- Fix(on-policy): fix the second order algorithms performance by [@Gaiejj](https://github.com/Gaiejj)  in PR [#147](https://github.com/PKU-MARL/omnisafe/pull/147) reviewed by [@friedmainfunction](https://github.com/friedmainfunction).
- Fix(rollout, exp_grid): fix logdir path conflict by [@muchvo](https://github.com/muchvo) in PR [#145](https://github.com/PKU-MARL/omnisafe/pull/145) reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@calico-1226](https://github.com/calico-1226).
- Feat(off-policy): add DDPG, TD3 SAC by [@Gaiejj](https://github.com/Gaiejj) in PR [#128](https://github.com/PKU-MARL/omnisafe/pull/128) reviewed by [@zmsn-2077](https://github.com/zmsn-2077).
- Feat: support policy evaluation by [@Gaiejj](https://github.com/Gaiejj) in PR [#137](https://github.com/PKU-MARL/omnisafe/pull/137) reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@Ruiyang Sun](https://github.com/rockmagma02).
- Fix: support new config for exp_grid by [@muchvo](https://github.com/muchvo) in PR [#142](https://github.com/PKU-MARL/omnisafe/pull/142) reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@Gaiejj](https://github.com/Gaiejj).
- Test: add more test case, and fix bugs by [@Ruiyang Sun](https://github.com/rockmagma02) in PR [#136](https://github.com/PKU-MARL/omnisafe/pull/136) reviewed by [@zmsn-2077](https://github.com/zmsn-2077), [@friedmainfunction](https://github.com/friedmainfunction) and [@Gaiejj](https://github.com/Gaiejj).
- Fix(ppo): fix entropy loss by [@Gaiejj](https://github.com/Gaiejj) in PR [#135](https://github.com/PKU-MARL/omnisafe/pull/135) reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@friedmainfunction](https://github.com/friedmainfunction).
- Chore: fix typo by [@1Asan](https://github.com/1Asan) in PR [#134](https://github.com/PKU-MARL/omnisafe/pull/134) reviewed by [@zmsn-2077](https://github.com/zmsn-2077).
- Fix(logger, wrapper): support csv file and velocity tasks by [@Gaiejj](https://github.com/Gaiejj) in PR [#131](https://github.com/PKU-MARL/omnisafe/pull/131) reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@friedmainfunction](https://github.com/friedmainfunction).
- Feat: update architecture of config.yaml by [@zmsn-2077](https://github.com/zmsn-2077) in PR [#126](https://github.com/PKU-MARL/omnisafe/pull/126) reviewed by [@friedmainfunction](https://github.com/friedmainfunction) and [@Gaiejj](https://github.com/Gaiejj).
- Chore: support num_thread setting by [@Gaiejj](https://github.com/Gaiejj) in PR [#124](https://github.com/PKU-MARL/omnisafe/pull/124) reviewed by [@Ruiyang Sun](https://github.com/rockmagma02).
- Fix(algo): fix no return in algo_wrapper::learn by [@Ruiyang Sun](https://github.com/rockmagma02) in PR [#122](https://github.com/PKU-MARL/omnisafe/pull/122) reviewed by [@zmsn-2077](https://github.com/zmsn-2077).
- Refactor: change architecture of omnisafe by [@Ruiyang Sun](https://github.com/rockmagma02) in PR [#121](https://github.com/PKU-MARL/omnisafe/pull/121) reviewed by [@zmsn-2077](https://github.com/zmsn-2077).
### 2023-02-27 ~ 2023-03-05
- Fix(off-policy): fix `action passing` by [@Gaiejj](https://github.com/Gaiejj) in PR [#119](https://github.com/PKU-MARL/omnisafe/pull/119) reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@friedmainfunction](https://github.com/friedmainfunction).
- Docs: update logo by [@Gaiejj](https://github.com/Gaiejj) in PR [#125](https://github.com/PKU-MARL/omnisafe/pull/125) reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@friedmainfunction](https://github.com/friedmainfunction).
- Fix(P3O): fix P3O performance by [@Gaiejj](https://github.com/Gaiejj) in PR [#123](https://github.com/PKU-MARL/omnisafe/pull/123) reviewed by [@zmsn-2077](https://github.com/zmsn-2077), [@Ruiyang Sun](https://github.com/rockmagma02) and [@calico-1226](https://github.com/calico-1226).
### 2023-02-13 ~ 2023-02-19
- Fix(evaluator): fix evaluator by [@Ruiyang Sun](https://github.com/rockmagma02) in PR [#117](https://github.com/PKU-MARL/omnisafe/pull/117) reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@calico-1226](https://github.com/calico-1226).
### 2023-02-06 ~ 2023-02-12
- Chore: modify `logo.png` and add `requirements.txt` by [@Ruiyang Sun](https://github.com/rockmagma02) in PR [#103](https://github.com/PKU-MARL/omnisafe/pull/103) reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@calico-1226](https://github.com/calico-1226).
- Build(env): delete local `safety-gymnaisum` dependence by [@Ruiyang Sun](https://github.com/rockmagma02) in PR [#102](https://github.com/PKU-MARL/omnisafe/pull/102) reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@calico-1226](https://github.com/calico-1226).
- Refactor(buffer): refactor `buffer` by [@Ruiyang Sun](https://github.com/rockmagma02) in PR [#101](https://github.com/PKU-MARL/omnisafe/pull/101) reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@friedmainfunction](https://github.com/friedmainfunction).
- Fix: fix tools by [@Gaiejj](https://github.com/Gaiejj) in PR [#100](https://github.com/PKU-MARL/omnisafe/pull/100) reviewed by [@calico-1226](https://github.com/calico-1226) and [@Ruiyang Sun](https://github.com/rockmagma02).
- Fix: fix algo wrapper by [@Gaiejj](https://github.com/Gaiejj) in PR [#99](https://github.com/PKU-MARL/omnisafe/pull/99) reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@Ruiyang Sun](https://github.com/rockmagma02).
- Refactor: clean the code by [@Gaiejj](https://github.com/Gaiejj) in PR [#97](https://github.com/PKU-MARL/omnisafe/pull/97) reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@Ruiyang Sun](https://github.com/rockmagma02).
### 2023-01-30 ~ 2023-02-05
- Chore: update linter settings by [@XuehaiPan](https://github.com/XuehaiPan).
- Chore: update ci  by [@Gaiejj](https://github.com/Gaiejj) in PR [#90](https://github.com/PKU-MARL/omnisafe/pull/90) reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@friedmainfunction](https://github.com/friedmainfunction).
- Chore: update yaml  by [@Gaiejj](https://github.com/Gaiejj) in PR [#92](https://github.com/PKU-MARL/omnisafe/pull/92) and [#93](https://github.com/PKU-MARL/omnisafe/pull/93) reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@friedmainfunction](https://github.com/friedmainfunction).
### 2023-01-23 ~ 2023-01-29
- Fix: fix seed setting  by [@Gaiejj](https://github.com/Gaiejj) in PR [#82](https://github.com/PKU-MARL/omnisafe/pull/82) reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@friedmainfunction](https://github.com/friedmainfunction).
- Refactor(objects): change object type into free_geom by [@muchvo](https://github.com/muchvo) in PR [#89](https://github.com/PKU-MARL/omnisafe/pull/89) reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@Gaiejj](https://github.com/Gaiejj).
- Chore: update algorithms configuration by [@Gaiejj](https://github.com/Gaiejj) in PR [#88](https://github.com/PKU-MARL/omnisafe/pull/88) reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@friedmainfunction](https://github.com/friedmainfunction).
- Feat: support cuda by [@Gaiejj](https://github.com/Gaiejj in PR [#86](https://github.com/PKU-MARL/omnisafe/pull/86) reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@friedmainfunction](https://github.com/friedmainfunction).
- Feat(render): add keyboard debug mode for some agents in all tasks by [@muchvo](https://github.com/muchvo) in PR [#83](https://github.com/PKU-MARL/omnisafe/pull/83) reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@Gaiejj](https://github.com/Gaiejj).
- Feat: add experiment grid by [@zmsn-2077](https://github.com/zmsn-2077) in PR [#84](https://github.com/PKU-MARL/omnisafe/pull/84) reviewed by [@friedmainfunction](https://github.com/friedmainfunction) and [@Gaiejj](https://github.com/Gaiejj).
### 2023-01-16 ~ 2023-01-22
- Feat(agents): add `ant` agent by [@muchvo](https://github.com/muchvo) in PR [#82](https://github.com/PKU-MARL/omnisafe/pull/82) reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@Gaiejj](https://github.com/Gaiejj).
- Refactor(safety-gymnaisum): `code decoupling` by [@muchvo](https://github.com/muchvo) in PR [#81](https://github.com/PKU-MARL/omnisafe/pull/81) reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@Gaiejj](https://github.com/Gaiejj).
- Feat: add new algorithm by [@Gaiejj](https://github.com/Gaiejj) in PR [#80](https://github.com/PKU-MARL/omnisafe/pull/80) reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@XuehaiPan](https://github.com/XuehaiPan).
### 2023-01-09 ~ 2023-01-15
- Refactor: change wrapper setting by [@Gaiejj](https://github.com/Gaiejj) in PR [#73](https://github.com/PKU-MARL/omnisafe/pull/73) reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@calico-1226](https://github.com/calico-1226).
- Feat: `vectorized` environment by [@Gaiejj](https://github.com/Gaiejj) in PR [#74](https://github.com/PKU-MARL/omnisafe/pull/74), reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@friedmainfunction](https://github.com/friedmainfunction).
### 2023-01-02 ~ 2023-01-08
- Feat(agents, tasks, Evaluator): support `circle012` and new agent `racecar`, update evaluator by [@muchvo](https://github.com/muchvo) in PR [#59](https://github.com/PKU-MARL/omnisafe/pull/59), reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@Gaiejj](https://github.com/Gaiejj).

### 2022-12-26 ~ 2023-01-01
- Refactor: enhanced model-based code, add `CAP` algorithm by [@hdadong](https://github.com/hdadong) in PR [#59](https://github.com/PKU-MARL/omnisafe/pull/59), reviewed by [@XuehaiPan](https://github.com/XuehaiPan) and [@zmsn-2077](https://github.com/zmsn-2077).
- Feat: support auto render as .mp4 videos, add examples and tests by [@muchvo](https://github.com/muchvo) in PR [#60](https://github.com/PKU-MARL/omnisafe/pull/60), reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@calico-1226](https://github.com/calico-1226).
- Fix(model-based): fix cap cost bug and lag beta value in cap.yaml by [@hdadong](https://github.com/hdadong) in PR [#62](https://github.com/PKU-MARL/omnisafe/pull/62), reviewed by [@zmsn-2077](https://github.com/zmsn-2077), [@calico-1226](https://github.com/calico-1226) and [@Gaiejj](https://github.com/Gaiejj).
- Fix(render): fix markers are not shown in the rgb array returned by env.render() by [@muchvo](https://github.com/muchvo) in PR [#61](https://github.com/PKU-MARL/omnisafe/pull/61), reviewed by [@Gaiejj](https://github.com/Gaiejj) and [@zmsn-2077](https://github.com/zmsn-2077).

### 2022-12-19 ~ 2022-12-25
- Feat(circle, run): support new tasks by [@muchvo](https://github.com/muchvo) in PR [#50](https://github.com/PKU-MARL/omnisafe/pull/50), reviewed by [@friedmainfunction](https://github.com/friedmainfunction) and [@Gaiejj](https://github.com/Gaiejj).
- in PR [#52](https://github.com/PKU-MARL/omnisafe/pull/52)
- Feat: add Makefile by [@XuehaiPan](https://github.com/XuehaiPan) in PR [#53](https://github.com/PKU-MARL/omnisafe/pull/53), reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@Gaiejj](https://github.com/Gaiejj).
- Fix: fix bug for namedtuple by [@Gaiejj](https://github.com/Gaiejj) in PR [#54](https://github.com/PKU-MARL/omnisafe/pull/54), reviewed by [@friedmainfunction](https://github.com/friedmainfunction) and [@zmsn-2077](https://github.com/zmsn-2077).
- Docs: fix spelling error by [@Gaiejj](https://github.com/Gaiejj) in PR [#56](https://github.com/PKU-MARL/omnisafe/pull/56), reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@XuehaiPan](https://github.com/XuehaiPan).


### 2022-12-12 ~ 2022-12-18
- Docs: retouch the formatting and add PPO docs for omnisafe by [@Gaiejj](https://github.com/Gaiejj) in PR [#40](https://github.com/PKU-MARL/omnisafe/pull/40), reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@calico-1226](https://github.com/calico-1226).
- Docs: add Lagrangian method documentation by [@Gaiejj](https://github.com/Gaiejj) in PR [#42](https://github.com/PKU-MARL/omnisafe/pull/42), reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@calico-1226](https://github.com/calico-1226)
- Refactor: change the details and yaml files of on policy algorithm by [@Gaiejj](https://github.com/Gaiejj) in PR [#41](https://github.com/PKU-MARL/omnisafe/pull/41), reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@calico-1226](https://github.com/calico-1226).
- Feat: add CUP algorithm by [@Gaiejj](https://github.com/Gaiejj) in PR [#43](https://github.com/PKU-MARL/omnisafe/pull/43), reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@friedmainfunction](https://github.com/friedmainfunction).
- Feat(wrapper): separated wrapper for different algorithmic environments by [@zmsn-2077](https://github.com/zmsn-2077) in PR [#44](https://github.com/PKU-MARL/omnisafe/pull/44), reviewed by [@friedmainfunction](https://github.com/friedmainfunction) and [@Gaiejj](https://github.com/Gaiejj).
- Refactor(README): show the implemented algorithms in more detail by [@zmsn-2077](https://github.com/zmsn-2077) in PR [#47](https://github.com/PKU-MARL/omnisafe/pull/47), reviewed by [@friedmainfunction](https://github.com/friedmainfunction) and [@Gaiejj](https://github.com/Gaiejj).
- Chore: rename files and enable pylint by [@muchvo](https://github.com/muchvo) in  PR [#39](https://github.com/PKU-MARL/omnisafe/pull/39), reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@Gaiejj](https://github.com/Gaiejj).
- Refactor: open pylint in pre-commit by [@zmsn-2077](https://github.com/zmsn-2077) in PR [#48](https://github.com/PKU-MARL/omnisafe/pull/48), reviewed by [@friedmainfunction](https://github.com/friedmainfunction) and [@Gaiejj](https://github.com/Gaiejj).

### 2022-12-05 ~ 2022-12-11

- Refactor: more OOP style code were used and made better code and file structure by [@muchvo](https://github.com/muchvo) in PR [#37](https://github.com/PKU-MARL/omnisafe/pull/37), reviewed by [@Gaiejj](https://github.com/Gaiejj) and [@zmsn-2077](https://github.com/zmsn-2077).
- Refactor: change the file layout of omnisafe by [@zmsn-2077](https://github.com/zmsn-2077) in PR [#35](https://github.com/PKU-MARL/omnisafe/pull/35), reviewed by [@Gaiejj](https://github.com/Gaiejj) and [@friedmainfunction](https://github.com/friedmainfunction).
- Docs: retouch the formatting and add links to the formula numbers by [@Gaiejj](https://github.com/Gaiejj) in PR [#31](https://github.com/PKU-MARL/omnisafe/pull/31), reviewed by [@XuehaiPan](https://github.com/XuehaiPan) and [@zmsn-2077](https://github.com/zmsn-2077).
- Fix(env_wrapper): fix warning caused by 'none' string default value by [@muchvo](https://github.com/muchvo) in PR [#30](https://github.com/PKU-MARL/omnisafe/pull/30), reviewed by [@XuehaiPan](https://github.com/XuehaiPan) and [@zmsn-2077](https://github.com/zmsn-2077).

### 2022-11-28 ~ 2022-12-04

- Chore(.github): update issue templates by [@XuehaiPan](https://github.com/XuehaiPan) in PR [#29](https://github.com/PKU-MARL/omnisafe/pull/29), reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@calico-1226](https://github.com/calico-1226).
- Docs: add trpo docs to omnisafe by [@Gaiejj](https://github.com/Gaiejj) in PR [#28](https://github.com/PKU-MARL/omnisafe/pull/28), reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@calico-1226](https://github.com/calico-1226).
- Refactor packaging by [@XuehaiPan](https://github.com/XuehaiPan) in PR [#20](https://github.com/PKU-MARL/omnisafe/pull/20), reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@hdadong](https://github.com/hdadong).
- Feat: add ddpg, clean some code, modify algo_wrapper in PR [#24](https://github.com/PKU-MARL/omnisafe/pull/24) by [@zmsn-2077](https://github.com/zmsn-2077), reviewd by [@Gaiejj](https://github.com/Gaiejj), [@XuehaiPan](https://github.com/XuehaiPan), and [@hdadong](https://github.com/hdadong).
- Add documentation of FOCOPS and PCPO by [@XuehaiPan](https://github.com/XuehaiPan) in [#21](https://github.com/PKU-MARL/omnisafe/pull/21), reviewed by [@Gaiejj](https://github.com/Gaiejj) and [@zmsn-2077](https://github.com/zmsn-2077).
- Add docs(focops,pcpo): add focops `docs` to omnisafe in PR [#19](https://github.com/PKU-MARL/omnisafe/pull/19) by [@Gaiejj](https://github.com/Gaiejj).

### 2022-11-20 ~ 2022-11-27

- Refactor some code in omnisafe, add `CHANGELOG.md`,` and del install.md and tutorial in PR [#16](https://github.com/PKU-MARL/omnisafe/pull/16) by [@zmsn-2077](https://github.com/zmsn-2077), reviewed by [@Gaiejj](https://github.com/Gaiejj) and [@hdadong](https://github.com/hdadong).
- Docs: add `PCPO` in omnisafe's **docs** and modify `CPO` by [@Gaiejj](https://github.com/Gaiejj) in [#9](https://github.com/PKU-MARL/omnisafe/pull/9), reviewed by [@zmsn-2077](https://github.com/zmsn-2077).
- Feat: add **Model-based Safe Algorithms:** `mbppolag`, `safeloop` by [@hdadong](https://github.com/hdadong) in [#12](https://github.com/PKU-MARL/omnisafe/pull/12), reviewed by [@zmsn-2077](https://github.com/zmsn-2077).
- Fix readme typo by [erjanmx](https://github.com/erjanmx) in PR [#13](https://github.com/PKU-MARL/omnisafe/pull/13), reviewed by [@zmsn-2077](https://github.com/zmsn-2077).
- Add render_mode: `human`, `rgb_array`, `depth_array` in safety-gymnasium: `safety_gym_v2` by [@zmsn-2077](https://github.com/zmsn-2077) in [#15](https://github.com/PKU-MARL/omnisafe/pull/15).
- Add .editorconfig and update license by [@XuehaiPan](https://github.com/XuehaiPan) in [#8](https://github.com/PKU-MARL/omnisafe/pull/8), and reviewed by [@zmsn-2077](https://github.com/zmsn-2077).
- Add `CPO` and `Intro` in omnisafe's **docs** by [@Gaiejj](https://github.com/Gaiejj) in PR [#7](https://github.com/PKU-MARL/omnisafe/pull/7), and reviewed by [@zmsn-2077](https://github.com/zmsn-2077).
- Fix ambiguous config yaml for algorithms by [@zmsn-2077](https://github.com/zmsn-2077) in PR [#6](https://github.com/PKU-MARL/omnisafe/pull/6), reviewed by [@Gaiejj](https://github.com/Gaiejj) and [@hdadong](https://github.com/hdadong).
- Add render mode and vision input in safety-gymnasium: `safety_gym_v2` by [@zmsn-2077](https://github.com/zmsn-2077) in PR [#5](https://github.com/PKU-MARL/omnisafe/pull/5), reviewed by [@hdadong](https://github.com/hdadong) and [@calico-1226](https://github.com/calico-1226).
- Fix vis `safety_gym_v2` with del the render_mode by [@zmsn-2077](https://github.com/zmsn-2077) in PR [#3](https://github.com/PKU-MARL/omnisafe/pull/3), reviewed by [@Gaiejj](https://github.com/Gaiejj) and [@hdadong](https://github.com/hdadong).
