import numpy as np
import torch
from typing import List
from data_generator.environments.flatland import FlatlandGenerator
from data_generator.groups import GroupProduct, CyclicGroup, generate_group, PermutationGroup

class CoupledSpeedGroup(GroupProduct):
    """
    GroupProduct where x-actions have a speed that depends on y-state
    Subclass of GroupProduct: overrides only transition()
    """
    def __init__(self, groups, x_group_idx=0, y_group_idx=1, speed_threshold=None):
        super().__init__(groups)
        self.x_group_idx = x_group_idx
        self.y_group_idx = y_group_idx
        # y position threshold
        self.speed_threshold = speed_threshold if speed_threshold is not None \
                               else groups[y_group_idx].n // 2

    def transition(self, idx, A):
        """
        Same as GroupProduct.transition but x-actions move by 1 or 2 steps
        depending on whether current y-state is below or above threshold.
        """
        if isinstance(A, torch.Tensor):
            A = A.numpy()
            was_tensor = True
        else:
            was_tensor = False

        idx = np.array(idx)
        B = len(idx)

        ka = self.A_to_a[A]       # [B, 2]: (subgroup_k, action_within_subgroup)
        k = ka[..., 0]            # which subgroup
        a = ka[..., 1]            # action within subgroup

        idxs = self.idx_to_idxs[idx]  # [B, n_groups]: per subgroup state indices

        for l in range(len(self.groups)):
            mask = k == l
            if not mask.any():
                continue

            if l == self.x_group_idx:
                # x action: speed depends on current y state
                y_states = idxs[mask, self.y_group_idx]
                fast = y_states >= self.speed_threshold  # select fast states

                # apply one step transition for slow states
                slow_mask = mask.copy()
                slow_mask[mask] = ~fast
                if slow_mask.any():
                    idxs[slow_mask, l] = self.groups[l].transition(
                        idxs[slow_mask, l], a[slow_mask])

                # apply two steps transition for fast states 
                fast_mask = mask.copy()
                fast_mask[mask] = fast
                if fast_mask.any():
                    tmp = self.groups[l].transition(idxs[fast_mask, l], a[fast_mask])
                    idxs[fast_mask, l] = self.groups[l].transition(tmp, a[fast_mask])
            else:
                idxs[mask, l] = self.groups[l].transition(idxs[mask, l], a[mask])

        result = self.idxs_to_idx[tuple(idxs.T)]
        return torch.from_numpy(result) if was_tensor else result


class FlatlandSpeedCoupled(FlatlandGenerator):
    """
    Flatland where x displacement depends on y position 
    dx = 1 if y < n_pos//2, dx = 2 if y >= n_pos//2
    """
    def __init__(self, n_pos, color_type, l=17, img_shape=[64, 64]):
        super().__init__(n_pos, color_type, l, img_shape)

        # replace the standard GroupProduct with the coupled version
        self.group = CoupledSpeedGroup(
            groups=[self.x_group, self.y_group, self.color_group],
            x_group_idx = 0,
            y_group_idx = 1,
            speed_threshold = n_pos // 2
        )

    def __repr__(self):
        return "flatland_speed_coupled"

    @property
    def specs(self):
        return {
            "n_pos": self.n_pos,
            "color_type": self.color_type,
            "l": self.l,
            "img_shape": self.img_shape,
        }

    def get_nfo(self):
        return {
            "x_dims": self.img_shape + [self.color_dim],
            "n_action": self.group.n_actions,
            "group": self.group.groups_list,
            "specs": self.specs,
            "environment": "flatland_speed_coupled",
        }



class InvertedActionGroup(GroupProduct):
    """
    GroupProduct where x actions are inverted when y is below a threshold
    """
    def __init__(self, groups, x_group_idx=0, y_group_idx=1, invert_threshold=None):
        super().__init__(groups)
        assert groups[x_group_idx].n_actions == 2
        self.x_group_idx = x_group_idx
        self.y_group_idx = y_group_idx
        # y position threshold: below -> x actions inverted, at/above -> normal
        self.invert_threshold = invert_threshold if invert_threshold is not None \
                                else groups[y_group_idx].n // 2

    def transition(self, idx, A):
        """
        Same as GroupProduct.transition but x actions are inverted (a -> 1-a)
        when the current y-state is below threshold
        """
        if isinstance(A, torch.Tensor):
            A = A.numpy()
            was_tensor = True
        else:
            was_tensor = False

        idx = np.array(idx)

        ka = self.A_to_a[A]       # [B, 2]: (subgroup_k, action_within_subgroup)
        k = ka[..., 0]            # which subgroup
        a = ka[..., 1]            # action within subgroup

        idxs = self.idx_to_idxs[idx]  # [B, n_groups]: per subgroup state indices

        for l in range(len(self.groups)):
            mask = k == l
            if not mask.any():
                continue

            if l == self.x_group_idx:
                y_states = idxs[mask, self.y_group_idx]
                inverted = y_states < self.invert_threshold  # True -> flip x+/x-

                a_l = a[mask].copy()
                a_l[inverted] = 1 - a_l[inverted]

                idxs[mask, l] = self.groups[l].transition(idxs[mask, l], a_l)
            else:
                idxs[mask, l] = self.groups[l].transition(idxs[mask, l], a[mask])

        result = self.idxs_to_idx[tuple(idxs.T)]
        return torch.from_numpy(result) if was_tensor else result


class FlatlandInvertedCoupled(FlatlandGenerator):
    """
    Flatland where x-actions are inverted (x+ <-> x-) when y < n_pos//2 
    """
    def __init__(self, n_pos, color_type, l=17, img_shape=[64, 64]):
        super().__init__(n_pos, color_type, l, img_shape)

        # Replace the standard GroupProduct with the inverted-action version
        self.group = InvertedActionGroup(
            groups=[self.x_group, self.y_group, self.color_group],
            x_group_idx=0,
            y_group_idx=1,
            invert_threshold=n_pos // 2
        )

    def __repr__(self):
        return "flatland_inverted_coupled"

    @property
    def specs(self):
        return {
            "n_pos": self.n_pos,
            "color_type": self.color_type,
            "l": self.l,
            "img_shape": self.img_shape,
        }

    def get_nfo(self):
        return {
            "x_dims": self.img_shape + [self.color_dim],
            "n_action": self.group.n_actions,
            "group": self.group.groups_list,
            "specs": self.specs,
            "environment": "flatland_inverted_coupled",
        }
