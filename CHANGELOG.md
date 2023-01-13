# Changelog

<!-- markdownlint-disable no-duplicate-header -->

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

------
## 2023-01-09 ~ 2023-01-15
- Refactor: change wrapper setting by [@Gaiejj](https://github.com/Gaiejj) in PR [#73](https://github.com/PKU-MARL/omnisafe/pull/73) reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@calico-1226](https://github.com/calico-1226).
- Feat: `vectorized` environment by [@Gaiejj](https://github.com/Gaiejj) in PR [#74](https://github.com/PKU-MARL/omnisafe/pull/74), reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [friedmainfunction](https://github.com/friedmainfunction).
## 2023-01-02 ~ 2023-01-08
- Feat(agents, tasks, Evaluator): support `circle012` and new agent `racecar`, update evaluator by [@muchvo](https://github.com/muchvo) in PR [#59](https://github.com/PKU-MARL/omnisafe/pull/59), reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@Gaiejj](https://github.com/Gaiejj).

## 2022-12-26 ~ 2023-01-01
- Refactor: enhanced model-based code, add `CAP` algorithm by [hdadong](https://github.com/hdadong) in PR [#59](https://github.com/PKU-MARL/omnisafe/pull/59), reviewed by [@XuehaiPan](https://github.com/XuehaiPan) and [@zmsn-2077](https://github.com/zmsn-2077).
- Feat: support auto render as .mp4 videos, add examples and tests by [@muchvo](https://github.com/muchvo) in PR [#60](https://github.com/PKU-MARL/omnisafe/pull/60), reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@calico-1226](https://github.com/calico-1226).
- Fix(model-based): fix cap cost bug and lag beta value in cap.yaml by [hdadong](https://github.com/hdadong) in PR [#62](https://github.com/PKU-MARL/omnisafe/pull/62), reviewed by [@zmsn-2077](https://github.com/zmsn-2077), [@calico-1226](https://github.com/calico-1226) and [@Gaiejj](https://github.com/Gaiejj).
- Fix(render): fix markers are not shown in the rgb array returned by env.render() by [@muchvo](https://github.com/muchvo) in PR [#61](https://github.com/PKU-MARL/omnisafe/pull/61), reviewed by [@Gaiejj](https://github.com/Gaiejj) and [@zmsn-2077](https://github.com/zmsn-2077).

## 2022-12-19 ~ 2022-12-25
- Feat(circle, run): support new tasks by [@muchvo](https://github.com/muchvo) in PR [#50](https://github.com/PKU-MARL/omnisafe/pull/50), reviewed by [friedmainfunction](https://github.com/friedmainfunction) and [@Gaiejj](https://github.com/Gaiejj).
- in PR [#52](https://github.com/PKU-MARL/omnisafe/pull/52)
- Feat: add Makefile by [@XuehaiPan](https://github.com/XuehaiPan) in PR [#53](https://github.com/PKU-MARL/omnisafe/pull/53), reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@Gaiejj](https://github.com/Gaiejj).
- Fix: fix bug for namedtuple by [@Gaiejj](https://github.com/Gaiejj) in PR [#54](https://github.com/PKU-MARL/omnisafe/pull/54), reviewed by [friedmainfunction](https://github.com/friedmainfunction) and [@zmsn-2077](https://github.com/zmsn-2077).
- Docs: fix spelling error by [@Gaiejj](https://github.com/Gaiejj) in PR [#56](https://github.com/PKU-MARL/omnisafe/pull/56), reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@XuehaiPan](https://github.com/XuehaiPan).


## 2022-12-12 ~ 2022-12-18
- Docs: retouch the formatting and add PPO docs for omnisafe by [@Gaiejj](https://github.com/Gaiejj) in PR [#40](https://github.com/PKU-MARL/omnisafe/pull/40), reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@calico-1226](https://github.com/calico-1226).
- Docs: add Lagrangian method documentation by [@Gaiejj](https://github.com/Gaiejj) in PR [#42](https://github.com/PKU-MARL/omnisafe/pull/42), reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@calico-1226](https://github.com/calico-1226)
- Refactor: change the details and yaml files of on policy algorithm by [@Gaiejj](https://github.com/Gaiejj) in PR [#41](https://github.com/PKU-MARL/omnisafe/pull/41), reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@calico-1226](https://github.com/calico-1226).
- Feat: add CUP algorithm by [@Gaiejj](https://github.com/Gaiejj) in PR [#43](https://github.com/PKU-MARL/omnisafe/pull/43), reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [friedmainfunction](https://github.com/friedmainfunction).
- Feat(wrapper): separated wrapper for different algorithmic environments by [@zmsn-2077](https://github.com/zmsn-2077) in PR [#44](https://github.com/PKU-MARL/omnisafe/pull/44), reviewed by [friedmainfunction](https://github.com/friedmainfunction) and [@Gaiejj](https://github.com/Gaiejj).
- Refactor(README): show the implemented algorithms in more detail by [@zmsn-2077](https://github.com/zmsn-2077) in PR [#47](https://github.com/PKU-MARL/omnisafe/pull/47), reviewed by [friedmainfunction](https://github.com/friedmainfunction) and [@Gaiejj](https://github.com/Gaiejj).
- Chore: rename files and enable pylint by [@muchvo](https://github.com/muchvo) in  PR [#39](https://github.com/PKU-MARL/omnisafe/pull/39), reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@Gaiejj](https://github.com/Gaiejj).
- Refactor: open pylint in pre-commit by [@zmsn-2077](https://github.com/zmsn-2077) in PR [#48](https://github.com/PKU-MARL/omnisafe/pull/48), reviewed by [friedmainfunction](https://github.com/friedmainfunction) and [@Gaiejj](https://github.com/Gaiejj).

## 2022-12-05 ~ 2022-12-11

- Refactor: more OOP style code were used and made better code and file structure by [@muchvo](https://github.com/muchvo) in PR [#37](https://github.com/PKU-MARL/omnisafe/pull/37), reviewed by [@Gaiejj](https://github.com/Gaiejj) and [@zmsn-2077](https://github.com/zmsn-2077).
- Refactor: change the file layout of omnisafe by [@zmsn-2077](https://github.com/zmsn-2077) in PR [#35](https://github.com/PKU-MARL/omnisafe/pull/35), reviewed by [@Gaiejj](https://github.com/Gaiejj) and [friedmainfunction](https://github.com/friedmainfunction).
- Docs: retouch the formatting and add links to the formula numbers by [@Gaiejj](https://github.com/Gaiejj) in PR [#31](https://github.com/PKU-MARL/omnisafe/pull/31), reviewed by [@XuehaiPan](https://github.com/XuehaiPan) and [@zmsn-2077](https://github.com/zmsn-2077).
- Fix(env_wrapper): fix warning caused by 'none' string default value by [@muchvo](https://github.com/muchvo) in PR [#30](https://github.com/PKU-MARL/omnisafe/pull/30), reviewed by [@XuehaiPan](https://github.com/XuehaiPan) and [@zmsn-2077](https://github.com/zmsn-2077).

## 2022-11-28 ~ 2022-12-04

- Chore(.github): update issue templates by [@XuehaiPan](https://github.com/XuehaiPan) in PR [#29](https://github.com/PKU-MARL/omnisafe/pull/29), reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@calico-1226](https://github.com/calico-1226).
- Docs: add trpo docs to omnisafe by [@Gaiejj](https://github.com/Gaiejj) in PR [#28](https://github.com/PKU-MARL/omnisafe/pull/28), reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@calico-1226](https://github.com/calico-1226).
- Refactor packaging by [@XuehaiPan](https://github.com/XuehaiPan) in PR [#20](https://github.com/PKU-MARL/omnisafe/pull/20), reviewed by [@zmsn-2077](https://github.com/zmsn-2077) and [@hdadong](https://github.com/hdadong).
- Feat: add ddpg, clean some code, modify algo_wrapper in PR [#24](https://github.com/PKU-MARL/omnisafe/pull/24) by [@zmsn-2077](https://github.com/zmsn-2077), reviewd by [@Gaiejj](https://github.com/Gaiejj), [@XuehaiPan](https://github.com/XuehaiPan), and [@hdadong](https://github.com/hdadong).
- Add documentation of FOCOPS and PCPO by [@XuehaiPan](https://github.com/XuehaiPan) in [#21](https://github.com/PKU-MARL/omnisafe/pull/21), reviewed by [@Gaiejj](https://github.com/Gaiejj) and [@zmsn-2077](https://github.com/zmsn-2077).
- Add docs(focops,pcpo): add focops `docs` to omnisafe in PR [#19](https://github.com/PKU-MARL/omnisafe/pull/19) by [@Gaiejj](https://github.com/Gaiejj).

## 2022-11-20 ~ 2022-11-27

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
