# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""A rollout reassembler that also says which rows share a token prefix."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import torch

from nemo_rl.data.packing.shared_prefix_metadata import (
    SHARED_PREFIX_PARENT,
    SHARED_PREFIX_PROMPT_LENGTHS,
    SHARED_PREFIX_SHARED_LEN,
    SHARED_PREFIX_SLOT,
    SHARED_PREFIX_TREE_COLUMNS,
    build_prefix_tree,
)
from nemo_rl.experience.rollout_reassembler import (
    FinalizedRollout,
    RolloutReassembler,
)


class SharedPrefixRolloutReassembler(RolloutReassembler):
    """Publish each group's prefix tree next to its rows.

    The rows themselves are unchanged, so rewards, advantages and loss masks
    are the same as with the default reassembler. The extra columns only say
    which rows start with the same tokens, and for how long. That holds for
    ordinary groups (siblings share the prompt) and for forest groups
    (compaction segments, and branch paths that share a parent call).

    Selection still happens in :mod:`nemo_rl.environments.call_selection`.
    This class runs after it and never changes which calls train.
    """

    extra_row_columns: ClassVar[tuple[str, ...]] = SHARED_PREFIX_TREE_COLUMNS

    def _row_extras(self, rows: Sequence[FinalizedRollout]) -> dict[str, torch.Tensor]:
        """Link each valid row to the earlier row it shares the most tokens with.

        This compares token IDs directly, so it is exact by definition. The
        selected calls' parent links could narrow the search for long groups;
        that is an optimization, not a different answer.
        """
        valid = [row.valid for row in rows]
        parents, shared_lens = build_prefix_tree([row.token_ids for row in rows], valid)
        return {
            # Placeholders keep prompt length 0, as before, so they run dense.
            SHARED_PREFIX_PROMPT_LENGTHS: torch.tensor(
                [int(row.prompt_len) if row.valid else 0 for row in rows],
                dtype=torch.long,
            ),
            SHARED_PREFIX_SLOT: torch.arange(len(rows), dtype=torch.long),
            SHARED_PREFIX_PARENT: torch.tensor(parents, dtype=torch.long),
            SHARED_PREFIX_SHARED_LEN: torch.tensor(shared_lens, dtype=torch.long),
        }
