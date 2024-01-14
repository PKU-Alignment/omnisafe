# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Test analyzing policies trained by exp-x with OmniSafe."""

import pytest

from omnisafe.common.statistics_tools import StatisticsTools


def test_statistics_tools():
    # just fill in the path in which experiment grid runs.
    path = './saved_source/test_statistics_tools'
    st = StatisticsTools()
    st.load_source(path)
    # just fill in the name of the parameter of which value you want to compare.
    # then you can specify the value of the parameter you want to compare,
    # or you can just specify how many values you want to compare in single graph at most,
    # and the function will automatically generate all possible combinations of the graph.
    # but the two mode can not be used at the same time.
    st.draw_graph('algo', None, 1)
    st.draw_graph('algo', ['PolicyGradient'], None)
    not_a_path = 'not_a_path'
    with pytest.raises(SystemExit):
        st.load_source(not_a_path)
