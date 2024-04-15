# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

<!-- insertion marker -->
## [v0.6.0](https://github.com/PKU-Alignment/omnisafe/releases/tag/v0.6.0) - 2024-04-15

<small>[Compare with v0.5.0](https://github.com/PKU-Alignment/omnisafe/compare/v0.5.0...v0.6.0)</small>

### Dependencies

- pin pandas version (#276) ([db34e2c](https://github.com/PKU-Alignment/omnisafe/commit/db34e2c34746f198c391509cf6e4c724058d2598) by Jiayi Zhou).

### Bug Fixes

- fix cuda initialization error in experiment grid (#315) ([63bdd34](https://github.com/PKU-Alignment/omnisafe/commit/63bdd3417ea415104aca1660619a23801cd3b957) by Jiayi Zhou).
- fix invalid tutorial links (#270) ([bea468e](https://github.com/PKU-Alignment/omnisafe/commit/bea468e2127bfc4865d88c5ef735970ab7789155) by Jinming Xu).
- Enable auto reset for MuJoCo envs (#269) ([b343750](https://github.com/PKU-Alignment/omnisafe/commit/b3437508bc3536e824f1ef9242ff2dc5380182e2) by Daniel Tan).
- correct the author list of PIDLag (#250) ([1a99b20](https://github.com/PKU-Alignment/omnisafe/commit/1a99b2009b779b7e4cbd87a280c592bc01589537) by Jinming Xu).

### Features

- support interface of environment customization (#310) ([efe7d9c](https://github.com/PKU-Alignment/omnisafe/commit/efe7d9cea96e743511e11b2056cbf05b5932364e) by Jiayi Zhou).
- support A3C parallel in multiple GPUs (#282) ([d55958a](https://github.com/PKU-Alignment/omnisafe/commit/d55958a011df7800f256452e07811832cd2524d2) by Jiayi Zhou).
- update config, benchmark results and code style (#280) ([9d943b6](https://github.com/PKU-Alignment/omnisafe/commit/9d943b6e15ade14da4a3c7380fad4de92c34b452) by Jiayi Zhou).
- update saute config and benchmarking results (#274) ([c575fd5](https://github.com/PKU-Alignment/omnisafe/commit/c575fd5044a3e042972018ddb4a427be58eec7bb) by Jiayi Zhou).
- support gymnasium style reset API (#266) ([80c2c23](https://github.com/PKU-Alignment/omnisafe/commit/80c2c23d0fc2e20f98778b21c33d0848b77567aa) by Jiayi Zhou).
- fix final_obsevation setting and support evaluation times configuration (#260) ([9e76d28](https://github.com/PKU-Alignment/omnisafe/commit/9e76d280f32a61519f28d0fc7d8a40e5125a30ba) by Jiayi Zhou).
- support new agents tasks and customizing environment guide (#256) ([7fcfe78](https://github.com/PKU-Alignment/omnisafe/commit/7fcfe789970bc5af063a8b8a12e036c411c5461c) by Jiayi Zhou).
- rebase CRPO from PPO to TRPO (#254) ([19f7fc7](https://github.com/PKU-Alignment/omnisafe/commit/19f7fc72e26750e138bc410e40c75cf9bb3a2199) by Jiayi Zhou).

### Style

- update pre-commit.yaml and fix ruff (#261) ([8702d79](https://github.com/PKU-Alignment/omnisafe/commit/8702d7980f5d6482b805b88037d3966d1eda6bee) by Jiayi Zhou).
- fix ruff check (#251) ([02376c7](https://github.com/PKU-Alignment/omnisafe/commit/02376c7a2c116c5629b4a3f69be2c2501f20b266) by Jiayi Zhou).

### Chore

- clear redundant distributed grad average (#253) ([af2951d](https://github.com/PKU-Alignment/omnisafe/commit/af2951d9209f7c5ce5a365d45664eb4ac6cb89ba) by Jiayi Zhou).

## [v0.5.0](https://github.com/PKU-Alignment/omnisafe/releases/tag/v0.5.0) - 2023-05-27

<small>[Compare with v0.4.0](https://github.com/PKU-Alignment/omnisafe/compare/v0.4.0...v0.5.0)</small>

### Bug Fixes

- fix adapter device and exp grid (#243) ([651cb0e](https://github.com/PKU-Alignment/omnisafe/commit/651cb0e14d598bdaba39fb79172408666f62151b) by Jiayi Zhou).

### Features

- support off-policy pid and update performance for navigation. (#245) ([18776f1](https://github.com/PKU-Alignment/omnisafe/commit/18776f178ed26f04c5bf9beb4ad493f88e1b6662) by Jiayi Zhou)
- improve test coverage and clear redundant code. (#238) ([dee636e](https://github.com/PKU-Alignment/omnisafe/commit/dee636e1bff5fff01d555c4fa7e8fe12671f0fec) by Jiayi Zhou).
- update benchmarks and provide configs for reproducing results (#236) ([3799d6d](https://github.com/PKU-Alignment/omnisafe/commit/3799d6d3fca4c8c94fed167d1a6cbd02d3b96992) by Jiayi Zhou).
- add `CODEOWNERS` and refine `ISSUE TEMPLATE` (#233) ([60b7ea7](https://github.com/PKU-Alignment/omnisafe/commit/60b7ea7aeb24fba83b1b6628a422dd537b74d0d4) by Jiaming Ji).
- add offline algorithms (#175) ([2a37bab](https://github.com/PKU-Alignment/omnisafe/commit/2a37babfa0b672e40e469df09d4418c747e6addb) by Ruiyang Sun).

### Style

- fix mypy and polish api docstring (#244) ([74e9527](https://github.com/PKU-Alignment/omnisafe/commit/74e9527cd2224a07f1fbcd9caaa61487118a049b) by Jiayi Zhou).

### Chore

- correct usage for `PROJECT_PATH` and `PROJECT_NAME` ([5085404](https://github.com/PKU-Alignment/omnisafe/commit/5085404ce540b909272738489a60d8edbc6d6b3c) by Xuehai Pan).
- update pre-commit hooks ([078608e](https://github.com/PKU-Alignment/omnisafe/commit/078608ef36086bfb26b25d6572988997d7b1dfb9) by Xuehai Pan).

## [v0.4.0](https://github.com/PKU-Alignment/omnisafe/releases/tag/v0.4.0) - 2023-05-04

<small>[Compare with v0.3.0](https://github.com/PKU-Alignment/omnisafe/compare/v0.3.0...v0.4.0)</small>

### Dependencies

- unpin `joblib` (#197) ([7f906f0](https://github.com/PKU-Alignment/omnisafe/commit/7f906f0def4fa78368dde9edb5816b73c4db23ef) by muchvo).

### Bug Fixes

- fix simmer performance and tune parameters (#220) ([a5b5335](https://github.com/PKU-Alignment/omnisafe/commit/a5b533512d2bccffa37f958cb8228e4784eea595) by Jiayi Zhou).
- enable `smooth` param in `Costs` when plotting (#208) ([295ec01](https://github.com/PKU-Alignment/omnisafe/commit/295ec01b4d4328e1cc5104ccdd9ed089093d6b7f) by muchvo).
- fix log when not update (#206) ([55b9db1](https://github.com/PKU-Alignment/omnisafe/commit/55b9db1a2a5ab0d059f4904d3de0530fcb64053f) by Jiayi Zhou).
- check duplicated parameters and values which are specified in experiment grid (#203) ([30d475d](https://github.com/PKU-Alignment/omnisafe/commit/30d475def23d19eaf8666336c305008edbc721c6) by muchvo).

### Features

- add model-based algorithms (#212) ([d65f37f](https://github.com/PKU-Alignment/omnisafe/commit/d65f37f7dde91b6784bbfe5abb39b433c9b246ab) by WeidongHuang).
- add `Dockerfile` and `codecov.yml` (#217) ([8bd253a](https://github.com/PKU-Alignment/omnisafe/commit/8bd253a9d5e8e105a31f02140cacfa4c7ca7998d) by Xuehai Pan).
- add algo and update test (#210) ([bc91d71](https://github.com/PKU-Alignment/omnisafe/commit/bc91d7128454bc6838be97182a76baa5a4233fde) by Jiayi Zhou).
- support saute rl and clean the code (#209) ([b0a0699](https://github.com/PKU-Alignment/omnisafe/commit/b0a0699c296e6fd2949de790f6d452dd5b5c58fa) by Jiaming Ji).
- support off-policy lag (#204) ([52aaf32](https://github.com/PKU-Alignment/omnisafe/commit/52aaf32b8be81996b491c360fe2c26499c958117) by Jiayi Zhou).

### Code Refactoring

- minor changes regarding documentation and dependencies (#219) ([15d5be6](https://github.com/PKU-Alignment/omnisafe/commit/15d5be6041deaaca0347ebbe510320a5e2c4b49d) by Jiaming Ji).
- refactor and simplify logger storage logic (#216) ([4dd6209](https://github.com/PKU-Alignment/omnisafe/commit/4dd62091fab33e3bc83459f5c468e5142de3131e) by Xuehai Pan).
- rename update cycle and refactor structure (#213) ([0eb08a3](https://github.com/PKU-Alignment/omnisafe/commit/0eb08a37e32c0f35b2b7954a0cfab815c92a3fce) by Jiayi Zhou).
- update hyper-parameters for first-order algorithms (#199) ([b2bcc84](https://github.com/PKU-Alignment/omnisafe/commit/b2bcc84e882131b631523f8db004b8039a16774f) by muchvo).
- condense top-level benchmarks (#198) ([a0a093b](https://github.com/PKU-Alignment/omnisafe/commit/a0a093baccb4e345b1694e726d9c32a0d2978210) by Jiaming Ji).

### Style

- support mypy checking and update docstring style (#221) ([c1c6993](https://github.com/PKU-Alignment/omnisafe/commit/c1c699338861e4a670db17a86b3f4eddf4a9e4f0) by Jiayi Zhou).
- prefer `utf-8` over `UTF-8` in code ([9449967](https://github.com/PKU-Alignment/omnisafe/commit/9449967fb1c61d21fbf05df72ae5bba6e0bc3b4f) by Xuehai Pan).
- fix grammar in README and normalize string in pyproject.toml ([d17aa77](https://github.com/PKU-Alignment/omnisafe/commit/d17aa77a9c622403548d2432bcb8739baa755045) by Xuehai Pan).

### Chore

- clean some trivial code (#214) ([e0b1852](https://github.com/PKU-Alignment/omnisafe/commit/e0b1852ab2b1c5de6eb7e027496704a944dd98df) by muchvo).
- update pre-commit hooks ([3dda988](https://github.com/PKU-Alignment/omnisafe/commit/3dda988bc6ffe67f759b54eaab7feb914a2fc9c4) by Xuehai Pan).
- track codecov for `dev` branch ([b7d3925](https://github.com/PKU-Alignment/omnisafe/commit/b7d3925ffd22f6e81d564777c37e12b76126b290) by Xuehai Pan).
- pre-commit autoupdate ([7f28646](https://github.com/PKU-Alignment/omnisafe/commit/7f28646e6c33146a3c8f412c0c5e3077b5b5c094) by Xuehai Pan).
- update benchmark performance for first-order algorithms (#215) ([65ed5ca](https://github.com/PKU-Alignment/omnisafe/commit/65ed5cab52a0258a6df1d5bfd8ec8805716964fa) by muchvo).
- [pre-commit.ci] autoupdate (#200) ([9393e27](https://github.com/PKU-Alignment/omnisafe/commit/9393e27b3724f76466a00b84f6fc078c0e2cdeac) by pre-commit-ci[bot]).
- upload tutorial (#201) ([b79bea7](https://github.com/PKU-Alignment/omnisafe/commit/b79bea7225082de46ef62b75fdcde7cae718d8ba) by muchvo).

## [v0.3.0](https://github.com/PKU-Alignment/omnisafe/releases/tag/v0.3.0) - 2023-04-01

<small>[Compare with v0.2.2](https://github.com/PKU-Alignment/omnisafe/compare/v0.2.2...v0.3.0)</small>

### Dependencies

- pin the version of pytorch (#184) ([3073b1a](https://github.com/PKU-Alignment/omnisafe/commit/3073b1a962f11df2692438e10c0e2e01052b9f5d) by muchvo).

### Bug Fixes

- fix file path problem when using gpu in experiment grid (#194) ([324d257](https://github.com/PKU-Alignment/omnisafe/commit/324d257012b418371b209c63b935f469392e95cf) by muchvo).

### Features

- update CLI for gpu and statistics tools (#192) ([858fc22](https://github.com/PKU-Alignment/omnisafe/commit/858fc221f6bf6377600d9e62fc087028308094cd) by muchvo).
- perfecting training workflow (#185) ([381b81c](https://github.com/PKU-Alignment/omnisafe/commit/381b81c6381dc32f0c4a221c535db1fc7c363e74) by muchvo).
- add `ruff` and `codespell` integration (#186) ([f070041](https://github.com/PKU-Alignment/omnisafe/commit/f07004171433849fe958d7024efdfc0327e10aa2) by Xuehai Pan).

## [v0.2.2](https://github.com/PKU-Alignment/omnisafe/releases/tag/v0.2.2) - 2023-03-28

<small>[Compare with v0.2.1](https://github.com/PKU-Alignment/omnisafe/compare/v0.2.1...v0.2.2)</small>

### Chore

- add MANIFEST.in (#182) ([f6b5aa4](https://github.com/PKU-Alignment/omnisafe/commit/f6b5aa4e97fcdaa6b4ed7b7293e92562dae77476) by muchvo).

## [v0.2.1](https://github.com/PKU-Alignment/omnisafe/releases/tag/v0.2.1) - 2023-03-27

<small>[Compare with v0.2.0](https://github.com/PKU-Alignment/omnisafe/compare/v0.2.0...v0.2.1)</small>

### Features

- support statistics tools for experiments launched by omnisafe (#157) ([eb5358e](https://github.com/PKU-Alignment/omnisafe/commit/eb5358efef74adcdd73eb8aeffb74ddd782b81fa) by muchvo).

## [v0.2.0](https://github.com/PKU-Alignment/omnisafe/releases/tag/v0.2.0) - 2023-03-26

<small>[Compare with v0.1.0](https://github.com/PKU-Alignment/omnisafe/compare/v0.1.0...v0.2.0)</small>

### Bug Fixes

- fix autoreset wrapper (#167) ([0e8114a](https://github.com/PKU-Alignment/omnisafe/commit/0e8114ab0e0cb713226cfd6af49651bba1115809) by r-y1).
- fix config assertion (#174) ([69eb514](https://github.com/PKU-Alignment/omnisafe/commit/69eb514380ffe335fed6896dcdcffa56780c2e0a) by Jiayi Zhou).
- fix the calculation of last state value (#164) ([7248752](https://github.com/PKU-Alignment/omnisafe/commit/72487526aad50e0540700320b7daa1bef07e8706) by muchvo).
- fix the calculation of last state value (#162) ([96e45f3](https://github.com/PKU-Alignment/omnisafe/commit/96e45f332ef0dcca505d85fe77533c3611208c3f) by Dtrc2207).

### Features

- support cuda (#163) ([e0ddf52](https://github.com/PKU-Alignment/omnisafe/commit/e0ddf527fa1477753eb5b3f2422bbb3b1d51cb16) by Jiayi Zhou).
- support command line interfaces for omnisafe (#144) ([d5e2814](https://github.com/PKU-Alignment/omnisafe/commit/d5e2814f2c2727337519d494d160f3532d955c93) by muchvo).

### Code Refactoring

- refactor the cuda setting (#176) ([3bf7660](https://github.com/PKU-Alignment/omnisafe/commit/3bf7660c7f498b0aa2632d9fd679551b2cc8551f) by Jiayi Zhou).

### Chore

- fix typo in readme (#172) ([e7cc549](https://github.com/PKU-Alignment/omnisafe/commit/e7cc549d0992913c7e2aa7876dcb2fed8480d6bd) by Ruiyang Sun).
- update github workflow (#156) ([e5bf84c](https://github.com/PKU-Alignment/omnisafe/commit/e5bf84c638f2f6947bb25faff4a582bfd1e506e1) by Jiayi Zhou).
- update yaml (#155) ([3014848](https://github.com/PKU-Alignment/omnisafe/commit/30148485a3b2fc27cbdd6bc3f242b08992ce192f) by Jiayi Zhou).

## [v0.1.0](https://github.com/PKU-Alignment/omnisafe/releases/tag/v0.1.0) - 2023-03-15

<small>[Compare with first commit](https://github.com/PKU-Alignment/omnisafe/compare/e2952161bbb22d0229f138df64e4c20d5757672e...v0.1.0)</small>

### Build

- delete local safety-gymnaisum dependence (#102) ([cd680f0](https://github.com/PKU-Alignment/omnisafe/commit/cd680f04d53e0f565d2a42b1a6a3b44e3cdde486) by Ruiyang Sun).

### Bug Fixes

- fix the second order algorithms performance (#147) ([5b960e8](https://github.com/PKU-Alignment/omnisafe/commit/5b960e800e9f38fc4622adb31af32995f2ec3b3a) by Jiayi Zhou).
- fix logdir path conflict (#145) ([268cb9d](https://github.com/PKU-Alignment/omnisafe/commit/268cb9d68d3e33706678afc52f806828787297a1) by muchvo).
- support new config for exp_grid (#142) ([380d8d0](https://github.com/PKU-Alignment/omnisafe/commit/380d8d0d4b1bb5ceb256358d31a924348b41a00d) by muchvo).
- fix entropy loss (#135) ([04e98ec](https://github.com/PKU-Alignment/omnisafe/commit/04e98ec000c1654decca945e787b12695a916b50) by Jiayi Zhou).
- fix entropy loss (#133) ([0ebae0a](https://github.com/PKU-Alignment/omnisafe/commit/0ebae0ac83d396a7ae789aa9c4d7c10435118aca) by Jiayi Zhou).
- support csv file and velocity tasks (#131) ([d661762](https://github.com/PKU-Alignment/omnisafe/commit/d6617622d2b683cb1b6988e86e18360225f11d9c) by Jiayi Zhou).
- fix action passing (#119) ([da3ecdf](https://github.com/PKU-Alignment/omnisafe/commit/da3ecdf722c1def0528257e76ab429b9b2a40194) by Jiayi Zhou).
- fix P3O performance (#123) ([53a8c1b](https://github.com/PKU-Alignment/omnisafe/commit/53a8c1bb564a5fb918c20cf0a455861bc20e5399) by Jiayi Zhou).
- fix no return in algo_wrapper::learn (#122) ([77807ee](https://github.com/PKU-Alignment/omnisafe/commit/77807eec54573fef14be4d7a748f82f19582d645) by Ruiyang Sun).
- fix evaluator (#117) ([323e3c6](https://github.com/PKU-Alignment/omnisafe/commit/323e3c6b306c1576cca04983fac74b31d1421ea7) by Jiayi Zhou).
- fix buffer bugs (#113) ([a527783](https://github.com/PKU-Alignment/omnisafe/commit/a527783474963f5e798823789728f296fa810c4c) by Ruiyang Sun).
- ensure the step of each epoch (#111) ([008337f](https://github.com/PKU-Alignment/omnisafe/commit/008337f95dfc46d59003834d965d7e8674be34f5) by Jiayi Zhou).
- fix tools (#100) ([72dfcc0](https://github.com/PKU-Alignment/omnisafe/commit/72dfcc00e635cd880e8874ac6d59fdd836a1437d) by Jiayi Zhou).
- fix algo wrapper (#99) ([75b8eac](https://github.com/PKU-Alignment/omnisafe/commit/75b8eac514d2f12d41c46f8f3ec140e3fb83735d) by Jiayi Zhou).
- fix seed setting (#91) ([af36ea4](https://github.com/PKU-Alignment/omnisafe/commit/af36ea45ed808625af154d71ef8ccdf8673a520f) by Jiayi Zhou).
- fix lagrange algorithms (#79) ([4b82ce0](https://github.com/PKU-Alignment/omnisafe/commit/4b82ce06dd36cbf1d2c5cb4c6136e6ccaf6dd2bb) by Jiayi Zhou).
- fix markers are not shown in the rgb array returned by env.render() (#61) ([b2966da](https://github.com/PKU-Alignment/omnisafe/commit/b2966da64564943531bf295fda9e59351d30f6d5) by muchvo).
- fix cap cost bug and lag beta value in cap.yaml (#62) ([6651298](https://github.com/PKU-Alignment/omnisafe/commit/66512987c85d74d124815e2b4edbfea05f89a1c0) by WeidongHuang).
- fix bug for namedtuple (#54) ([199d3e4](https://github.com/PKU-Alignment/omnisafe/commit/199d3e4a10247c7ecd5e0f8cd57c848d8e40d059) by Jiayi Zhou).
- fix warning caused by 'none' value in the __init__(). (#30) ([65ef16e](https://github.com/PKU-Alignment/omnisafe/commit/65ef16ef7bb05b91416c79f51ac6dd6389c312fc) by muchvo).
- clean the config yaml (#6) ([ea4758a](https://github.com/PKU-Alignment/omnisafe/commit/ea4758ac11418a96ba6771ae397308d08cb5c0ea) by zmsn-2077).
- del render_model (#3) ([3536b83](https://github.com/PKU-Alignment/omnisafe/commit/3536b836fc9e7a1749e20a7f8db14781d86be14f) by zmsn-2077).

### Features

- add DDPG, TD3 SAC (#128) ([41c5621](https://github.com/PKU-Alignment/omnisafe/commit/41c56216087c1b8b6fcb17cb023cb71aa756b07a) by Jiayi Zhou).
- support policy evaluation (#137) ([3de924f](https://github.com/PKU-Alignment/omnisafe/commit/3de924f6858b24d2b194ede9566f1e70645bcf89) by Jiayi Zhou).
- update architecture of config.yaml (#126) ([576e2c7](https://github.com/PKU-Alignment/omnisafe/commit/576e2c72cf9cec4c831a61884a4dbc56cfb9f21b) by zmsn-2077).
- modify config class (#107) ([aa68319](https://github.com/PKU-Alignment/omnisafe/commit/aa683191bb947b9ad1f881a3326b8724a17b4dd2) by Ruiyang Sun).
- support cuda (#86) ([334ab33](https://github.com/PKU-Alignment/omnisafe/commit/334ab33d36dbe1ee9e5f06211287a78273f1ec4c) by Jiayi Zhou).
- add keyboard debug mode for some agents in all tasks (#83) ([42d0b79](https://github.com/PKU-Alignment/omnisafe/commit/42d0b79c01177d3ecbfc40853f5ddf0d3c147355) by muchvo).
- add experiment grid (#84) ([42ca77d](https://github.com/PKU-Alignment/omnisafe/commit/42ca77dc06950650d8c8b14caea0464d7d8b00c7) by zmsn-2077).
- add ant agent (#82) ([6cdf7e3](https://github.com/PKU-Alignment/omnisafe/commit/6cdf7e30731a7e878f4ec39ae8e68072605d599b) by muchvo).
- add new algorithm (#80) ([5e3a607](https://github.com/PKU-Alignment/omnisafe/commit/5e3a607b15c726926f228728d144881edf26f6db) by Jiayi Zhou).
- support velocity tasks (#68) ([0b555b6](https://github.com/PKU-Alignment/omnisafe/commit/0b555b686a472c9feaa83901857f7028669d6ceb) by muchvo).
- vectorized environment (#74) ([ff9e332](https://github.com/PKU-Alignment/omnisafe/commit/ff9e332b38531ab0f4cab2c292f0613680407192) by Jiayi Zhou).
- support circle012 and new agent racecar, update evaluator (#66) ([babfaac](https://github.com/PKU-Alignment/omnisafe/commit/babfaac959be44ba8f84a83037bfbfd6b6f7f165) by muchvo).
- support auto render as .mp4 videos, add examples and tests (#60) ([1287695](https://github.com/PKU-Alignment/omnisafe/commit/128769573b4a1567d84685f06e081d7b3973f1c9) by muchvo).
- add `Makefile` (#53) ([e0aeec0](https://github.com/PKU-Alignment/omnisafe/commit/e0aeec04b742b8a9f362cd9d118a5297b7f91299) by Xuehai Pan).
- add new algorithms (#52) ([9747985](https://github.com/PKU-Alignment/omnisafe/commit/9747985409260a1eda8b619ee8bc1a1fb92af0b3) by Jiayi Zhou).
- support new tasks. (#50) ([fb702d8](https://github.com/PKU-Alignment/omnisafe/commit/fb702d8f2d71e362fe0d938043c8c773f69ff427) by muchvo).
- separated wrapper for different algorithmic environments (#44) ([d1e171e](https://github.com/PKU-Alignment/omnisafe/commit/d1e171e14bb5c904e28294b7adcc0697fc32bdff) by zmsn-2077).
- add CUP algorithm (#43) ([d4cd28b](https://github.com/PKU-Alignment/omnisafe/commit/d4cd28b9d1e542f420776208332e057cc017fde9) by 周嘉懿).
- support rgb_array and add metadata (#15) ([5a057b5](https://github.com/PKU-Alignment/omnisafe/commit/5a057b52898935e3ddcb40fdfcfc1d48aecb4190) by zmsn-2077).

### Code Refactoring

- change architecture of omnisafe (#121) ([555acbb](https://github.com/PKU-Alignment/omnisafe/commit/555acbbf10643420f13b0d6197a54e55a9bf47ce) by Ruiyang Sun).
- refactor logger (#115) ([96d609e](https://github.com/PKU-Alignment/omnisafe/commit/96d609e0884c9bf55a0bd078d794070d9ea7fe6b) by Ruiyang Sun).
- refactor buffer (#101) ([42282d6](https://github.com/PKU-Alignment/omnisafe/commit/42282d6c2879b0f16ef5a3e3e2e7e003f57e0691) by Ruiyang Sun).
- clean the code (#97) ([7b4860b](https://github.com/PKU-Alignment/omnisafe/commit/7b4860b1eaaec81883990b0c135e9e4cdbad78f4) by Jiayi Zhou).
- change object type into free_geom (#89) ([105e123](https://github.com/PKU-Alignment/omnisafe/commit/105e123eb296b2136281b8c462d7e86fdaa6e2d9) by muchvo).
- code decoupling (#81) ([bfdf458](https://github.com/PKU-Alignment/omnisafe/commit/bfdf458c1d28e2f9e8b04a7780f9eea559bbd390) by muchvo).
- update CHANGELOG.md (#77) ([88aabe6](https://github.com/PKU-Alignment/omnisafe/commit/88aabe692ef751967a777b4f399044a994308342) by zmsn-2077).
- change wrapper setting (#73) ([e975b7d](https://github.com/PKU-Alignment/omnisafe/commit/e975b7d675d2ad8e0446b5857afd7601b4ad72a5) by Jiayi Zhou).
- enhanced model-based code, add CAP algorithm (#59) ([7874ab6](https://github.com/PKU-Alignment/omnisafe/commit/7874ab6b663b6854f0b9e8bbd6655446a321ddc9) by WeidongHuang).
- open pylint in pre-commit (#48) ([edd452b](https://github.com/PKU-Alignment/omnisafe/commit/edd452ba923f248f9d3e095ee2bc2a5b28b90d85) by zmsn-2077).
- show the implemented algorithms in more detail (#47) ([cdd92f2](https://github.com/PKU-Alignment/omnisafe/commit/cdd92f24b5e46931db679c05f15d1e4371b87d6c) by zmsn-2077).
- change the details and yaml files of on policy algorithm. (#41) ([2c8dbb3](https://github.com/PKU-Alignment/omnisafe/commit/2c8dbb36840b100a453901a7a4a851e66e0a0e1a) by 周嘉懿).
- More OOP style code were used and made better code and file structure. (#37) ([4cc94a2](https://github.com/PKU-Alignment/omnisafe/commit/4cc94a298be82296250f0a911b8113be379a0439) by muchvo).
- change the file layout of omnisafe (#35) ([72abc6e](https://github.com/PKU-Alignment/omnisafe/commit/72abc6efdc49d60cad05236602aa5d2efb3133f3) by zmsn-2077).
- refactor packaging (#20) ([7beb606](https://github.com/PKU-Alignment/omnisafe/commit/7beb606562e64a4544d4a570373bb9ee684559ac) by Xuehai Pan).

### Chore

- update benchmark performance for first-order algorithms (#148) ([0802117](https://github.com/PKU-Alignment/omnisafe/commit/0802117a818d3bbc869a031b14b1cfeb360b40c8) by muchvo).
- fix typo. (#134) ([515ca4d](https://github.com/PKU-Alignment/omnisafe/commit/515ca4d75b25d0ec5893637f9ed008220a513dac) by 1Asan).
- support num_thread setting (#124) ([a88b03b](https://github.com/PKU-Alignment/omnisafe/commit/a88b03b09c758ba56b6b5983b06f94d465403745) by Jiayi Zhou).
- update logo.png and dependency (#116) ([42ed91e](https://github.com/PKU-Alignment/omnisafe/commit/42ed91e869617a2e6a5a24432352d03614eb9c74) by zmsn-2077).
- update ppo.yaml (#112) ([97fcd2e](https://github.com/PKU-Alignment/omnisafe/commit/97fcd2eb32236baca757b4c1451143d2396068d1) by Jiayi Zhou).
- workaround upstream torch bug (#109) ([2e13bd6](https://github.com/PKU-Alignment/omnisafe/commit/2e13bd6f4e0f4b318df16be4349080c18013b91f) by zmsn-2077).
- update changelog and readme (#106) ([bf5130e](https://github.com/PKU-Alignment/omnisafe/commit/bf5130e2daf349188aafb5541d7830e525c15b1d) by zmsn-2077).
- modify logo.png and add requirements.txt (#103) ([74ef4bb](https://github.com/PKU-Alignment/omnisafe/commit/74ef4bbbc2dc1090590532ae1610fd699aaca53d) by Ruiyang Sun).
- update linter settings ([9c6cee5](https://github.com/PKU-Alignment/omnisafe/commit/9c6cee54e87cb7f7a2f1ace9f07b706739a2ecdc) by Xuehai Pan).
- update yaml (#93) ([cc6f4c9](https://github.com/PKU-Alignment/omnisafe/commit/cc6f4c9736777aa1a6e4ffbee07c6efce7879e60) by Jiayi Zhou).
- update ci (#90) ([f0b2324](https://github.com/PKU-Alignment/omnisafe/commit/f0b2324396b58d379e5df5fa52c4caca6928c4b1) by Jiayi Zhou).
- update yaml (#92) ([f88c23f](https://github.com/PKU-Alignment/omnisafe/commit/f88c23fb683786987a78cef8443feaae68c604ab) by Jiayi Zhou).
- update algorithms configuration (#88) ([1d39005](https://github.com/PKU-Alignment/omnisafe/commit/1d390051f45e55235402b680e970a529f1362487) by Jiayi Zhou).
- update setup.py for safety-gymnasium (#78) ([c1a9171](https://github.com/PKU-Alignment/omnisafe/commit/c1a9171f61a0ed9862bbf939f73e921baf3961e0) by muchvo).
- rename files and enable pylint. (#39) ([547517e](https://github.com/PKU-Alignment/omnisafe/commit/547517e5c0e5679084c026db1965e36a6a367303) by muchvo).
- update issue templates ([5d54fca](https://github.com/PKU-Alignment/omnisafe/commit/5d54fcad7eeb11ea841b8b650cf7a97446129436) by Xuehai Pan).
- update issue templates (#29) ([95522dc](https://github.com/PKU-Alignment/omnisafe/commit/95522dc4d4b9c9c271d202bfbf196cd1004e86fb) by Xuehai Pan).
- add CHANGELOG.md and update some statement. (#16) ([c616235](https://github.com/PKU-Alignment/omnisafe/commit/c616235f029d403d0f6a670809b16f98cbf533c4) by zmsn-2077).
- add .editorconfig and update license (#8) ([9452e35](https://github.com/PKU-Alignment/omnisafe/commit/9452e35197c37a449170ce0689bcdffef4954302) by Xuehai Pan).
