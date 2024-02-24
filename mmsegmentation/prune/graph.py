from collections import defaultdict
from typing import Dict, List, Union
import numpy as np
import torch

COUPLING_OPS = [
    "AddBackward0",
    "BmmBackward0",
]

STOP_OPS = [
    "ConvolutionBackward0",
    "AddmmBackward0",
    "MmBackward0",
    "CudnnBatchNormBackward0",
    "NativeLayerNormBackward0",
    "NativeGroupNormBackward0",
]


def flatten(list: List[List[any]]) -> List[any]:
    return [val for subl in list for val in subl]


class Graph:
    def __init__(self) -> None:
        self.nodes = dict()
        self.in_edges = defaultdict(list)
        self.out_edges = defaultdict(list)
        self.node_attr: dict[str, dict[str, Union[tuple, int]]] = dict()

    def add_edge(self, n1, n2):
        if type(n1) is not int:
            n1 = id(n1)
        if type(n2) is not int:
            n2 = id(n2)
        self.in_edges[n2].append(n1)
        self.out_edges[n1].append(n2)

    def add_node(self, node, name):
        if type(node) is not int:
            node = id(node)
        if node not in self.nodes:
            self.nodes[node] = name

    def add_attr_to_node(self, node, attr):
        if type(node) is not int:
            node = id(node)
        self.node_attr[node] = attr

    def get_node_in_edges(self, node):
        return self.in_edges[node]

    def get_node_out_edges(self, node):
        return self.out_edges[node]

    def get_nodes_by_name(self, name):
        return [node for node, node_name in self.nodes.items() if node_name == name]

    def trace_node(self, node, max_depth=3):
        def trace(node, prefix, depth):
            print(prefix + self.get_node_name(node))
            if depth <= 0:
                return
            for follow in self.get_node_out_edges(node):
                trace(follow, "  " + prefix, depth - 1)

        trace(node, prefix="| ", depth=max_depth)

    def get_named_nodes(self, return_names=False):
        if return_names:
            return [
                (node, node_name)
                for node, node_name in self.nodes.items()
                if node_name != ""
            ]
        else:
            return [node for node, node_name in self.nodes.items() if node_name != ""]

    def get_node_name(self, node):
        if type(node) is list or type(node) is set:
            return [self.nodes[n] for n in node]
        else:
            return self.nodes[node]

    def get_pruning_groups(
        self,
        params: Dict[int, str],
        model: torch.nn.Module,
        exclude_params=[
            "decode_head.conv_seg.weight",
            "decode_head.conv_seg.bias",
            "head.fc.weight",
            "head.fc.bias",
            "head.layers.head.bias",
            "head.layers.head.weight",
            "decode_head.mask_norm.weight",
            "decode_head.mask_norm.bias",
        ],
    ):
        """
        Returns a list of sets which contain all output->input conditioned groups.
        This means for each node (e.g. convolution) the corresponding groups are returned which,
        when pruned on their output channel, require the node to remove the corresponding input channel.

        Exclude_parameters specifies ops which should not be added to the output groups,
        i.e. they will not be pruned.
        This is mostly used to exclude the output layers of the model.
        """

        def trace_to_stop_or_cond_op(op: int):
            """
            Accumulates all stop or coupling ops amongst the outputs of the given op.
            """
            # Found stop op -> Add to stop op
            if self.get_node_name(op) in STOP_OPS:
                return set([op]), None

            # Found cond op -> Add to output group
            if self.get_node_name(op) in COUPLING_OPS:
                return None, set([op])

            s_ops, c_ops = set(), set()
            for outp in self.get_node_out_edges(op):
                s_op, c_op = trace_to_stop_or_cond_op(outp)
                if s_op is not None:
                    s_ops.update(s_op)
                if c_op is not None:
                    c_ops.update(c_op)
            return s_ops, c_ops

        param2s_op: Dict[int, int] = {}
        param2c_op: Dict[int, int] = {}
        s_op2params = defaultdict(list)
        c_op2params = defaultdict(list)
        # Match parameters to their respective stop op or coupling op
        for p_name in params.values():
            p_node = self.get_nodes_by_name(p_name)
            if len(p_node) == 1:
                s_ops, c_ops = trace_to_stop_or_cond_op(
                    self.get_nodes_by_name(p_name)[0]
                )
                if len(s_ops) > 0:
                    for s_op in s_ops:
                        s_op2params[s_op].append(p_node[0])
                        param2s_op[p_node[0]] = s_op
                if len(c_ops) > 0:
                    for c_op in c_ops:
                        c_op2params[c_op].append(p_node[0])
                        param2c_op[p_node[0]] = c_op

        coupling_ops = {}
        following_stop_ops = {}

        # Accumulate all coupling ops a param encounters on its way to it's stop ops
        # and also accumulate them
        for p_name in params.values():
            s_ops, c_ops = [], []
            p_node = self.get_nodes_by_name(p_name)
            if len(p_node) != 1:
                continue
            p_node = p_node[0]
            if p_node in param2s_op:
                s_op = param2s_op[p_node]
                node_name = self.get_node_name(s_op)
                if node_name in [
                    "CudnnBatchNormBackward0",
                    "ConvolutionBackward0",
                ]:
                    if (
                        node_name == "ConvolutionBackward0"
                        and self.node_attr[s_op]["groups"] != 1
                    ):
                        # Dwconv, add itself to following convs since input and output dimensions are related
                        c_ops.append((s_op, (1,)))
                        s_ops.append((s_op, (1,)))
                    # Parameter channel dimension of conv/bn weight always is 1
                    dim = (1,)
                elif node_name == "AddmmBackward0":
                    # Linear layer, channel dimension is 1
                    dim = (1,)
                else:
                    # For other ops, channel dimension of the parameter is usually the last dimension
                    dim = (len(self.node_attr[s_op]["output_shapes"][0]) - 1,)
                for outp in self.get_node_out_edges(s_op):
                    # Actually trace all stop and coupling ops along with the index
                    # the channel dimension has at the found op for the given parameter
                    if p_name == "backbone.pag_1.f_p.bn.weight":
                        s, c = self.trace_dim_stop_and_coupling_ops(
                            outp, dim=dim, print=print
                        )
                    else:
                        s, c = self.trace_dim_stop_and_coupling_ops(outp, dim=dim)
                    s_ops.extend(s)
                    c_ops.extend(c)

            for s_op in s_ops:
                # Register every stop op and the found channel dimension for the parameter
                # Each element s_op is a tuple containing (found_stop_op, dim)
                if p_name not in following_stop_ops:
                    following_stop_ops[p_name] = defaultdict(set)
                for dim in s_op[1]:
                    following_stop_ops[p_name][dim].update(
                        set(self.get_node_name(s_op2params[s_op[0]]))
                    )
            for c_op in c_ops:
                # Register every coupling op and the found channel dimension for the parameter
                # Each element s_op is a tuple containing (found_coupling_op, dim)
                if c_op[0] not in coupling_ops:
                    coupling_ops[c_op[0]] = defaultdict(list)
                for dim in c_op[1]:
                    coupling_ops[c_op[0]][dim].append(p_name)

        def trace_back_to_param(op, p_node, coupled_dim, skip=False):
            # Helper function which traces the channel dimension backwards from a stop op
            # to the parameter it belongs to.
            op_name = self.get_node_name(op)
            if op_name == "TBackward0":
                # Transpose - Swap dims
                coupled_dim = coupled_dim - 1
            if op_name == self.get_node_name(p_node):
                return coupled_dim
            if op_name in STOP_OPS:
                if skip:
                    skip = False
                else:
                    return None

            for inp in self.get_node_in_edges(op):
                dim = trace_back_to_param(inp, p_node, coupled_dim)
                if dim is not None:
                    return dim
            return None

        # Accumulate coupling_ops, stop_ops for missing parameters.
        # Often the case with e.g. positional embeddings
        # This process is similar to the above
        for p_name in params.values():
            p_node = self.get_nodes_by_name(p_name)
            if len(p_node) != 1:
                continue
            p_node = p_node[0]
            if p_node in param2c_op:
                c_op = param2c_op[p_node]
                traced_dim = 2
                coupling_ops[c_op][traced_dim].append(p_name)
                s_ops, c_ops = [], []
                # Make sure we do not have already added this parameter
                if not p_node in param2s_op:
                    for outp in self.get_node_out_edges(p_node):
                        s, c = self.trace_dim_stop_and_coupling_ops(
                            outp, dim=(traced_dim,)
                        )
                        s_ops.extend(s)
                        c_ops.extend(c)
                    for s_op in s_ops:
                        if p_name not in following_stop_ops:
                            following_stop_ops[p_name] = defaultdict(list)
                        for dim in s_op[1]:
                            following_stop_ops[p_name][dim].extend(
                                self.get_node_name(s_op2params[s_op[0]])
                            )
                    for c_op in c_ops:
                        if c_op[0] not in coupling_ops:
                            coupling_ops[c_op[0]] = defaultdict(list)
                        for dim in c_op[1]:
                            coupling_ops[c_op[0]][dim].append(p_name)

        # List which will contain all coupling groups sorted by the number
        # of parameters they contain. This makes merging them easier
        c_groups_sorted: list[set] = []

        # Flatten
        for c_op in coupling_ops.items():
            for dim, p in c_op[1].items():
                c_groups_sorted.append((c_op[0], dim, p))
                sops = defaultdict(int)
                for pi in p:
                    pi_node = self.get_nodes_by_name(pi)[0]
                    if pi_node in param2s_op:
                        sops[param2s_op[pi_node]] += 1

        added_params = set(flatten([a[2] for a in c_groups_sorted]))

        missing_parameters = set(params.values()).difference(added_params)
        missing_parameters = set(
            [
                p
                for p in missing_parameters
                if len(self.get_nodes_by_name(p)) == 1 and p not in exclude_params
            ]
        )
        # Add all remaining missing parameters of the model to the coupling groups as
        # groups only containing themselves
        for p in missing_parameters:
            p_node = self.get_nodes_by_name(p)[0]
            s_op = param2s_op[p_node]
            node_name = self.get_node_name(s_op)
            if node_name in [
                "CudnnBatchNormBackward0",
                "ConvolutionBackward0",
            ]:
                dim = 1
            else:
                # Last dimension
                dim = len(self.node_attr[s_op]["output_shapes"][0]) - 1
            traced_dim = trace_back_to_param(
                s_op, self.get_nodes_by_name(p)[0], dim, skip=True
            )
            # Set None as coupling group since it's a singular parameter
            c_groups_sorted.append((None, traced_dim, [p]))
        c_groups_sorted.sort(key=lambda x: len(x[2]), reverse=True)
        print("Num coupling groups including missing:", len(c_groups_sorted))
        c_groups: list[list] = []

        # Find all batch norm weight parameter names
        norm_weights = []
        for node in flatten(
            [
                self.get_nodes_by_name(norm)
                for norm in [
                    "CudnnBatchNormBackward0",
                    "NativeLayerNormBackward0",
                    "NativeGroupNormBackward0",
                ]
            ]
        ):
            norm_weights.extend(self.get_node_name(s_op2params[node]))

        # Trim redundant coupling groups by deleting true subgroups of larger
        # coupling groups or by merging coupling groups if they have a non-empty intersection
        for c_group in c_groups_sorted:
            add = True
            # # If a group only consists of a single parameter that is belonging
            # # to a batch norm, we can ignore that group since the will be contained
            # # in the following ops of the corresponding convolution
            for idx, already_added in enumerate(c_groups):
                op_name = (
                    self.get_node_name(c_group[0]) if c_group[0] is not None else ""
                )
                other_name = (
                    self.get_node_name(already_added[0])
                    if already_added[0] is not None
                    else ""
                )
                # In case of BmmBackward0 we can merge the groups regardless of channel indexes
                # this is done to prevent duplicate output groups
                if op_name == "BmmBackward0" and op_name == other_name:
                    if set(c_group[2]).issubset(already_added[2]):
                        add = False
                        break
                    if not set(c_group[2]).isdisjoint(set(already_added[2])):
                        add = False
                        c_groups[idx][2].extend(c_group[2])
                        break
                # Only merge on matching dimensions
                # If it's a true subset, do not add
                if set(c_group[2]).issubset(set(already_added[2])):
                    add = False
                    break
                # If it's a intersection, merge them
                if not set(c_group[2]).isdisjoint(set(already_added[2])):
                    if c_group[1] == already_added[1]:
                        add = False
                        c_groups[idx][2].extend(c_group[2])
                        break
            if add:
                c_groups.append(c_group)
            # print()

        output_groups: list[list] = []
        following_groups: list[list] = []

        # Accumulating following groups
        for idx, g in enumerate(c_groups):
            c_op, dim, gi = g
            dim_wise_follow_ops = defaultdict(set)
            for fgi in gi:
                for d, ops in following_stop_ops[fgi].items():
                    dim_wise_follow_ops[d].update(set(ops))
                if fgi in norm_weights:
                    dim_wise_follow_ops[1].update(set([fgi]))
            output_groups.append((dim, set(gi)))
            following_groups.append(dim_wise_follow_ops)

        # Progressively merge following groups which contain overlapping parameters
        # This is done to ensure consistency
        idx = 0
        ln = len(following_groups)
        while idx < ln:
            fops = following_groups[idx]
            overlaps = []
            for j, other in enumerate(following_groups):
                if j == idx:
                    continue
                for dim, items in fops.items():
                    if dim in other:
                        ints = set(items).intersection(other[dim])
                        if len(ints) > 0:
                            overlaps.append((j, dim))
                            fops[dim].update(other[dim])
            # If there are overlaps, merge them into the larger group (specified by idx) and delete the former groups
            if len(overlaps) > 0:
                n_og, n_fg = output_groups[idx], fops
                len_deleted = 0
                # Add all operations of the overlapping groups to the largest group
                for j, dim in overlaps:
                    n_og[1].update(output_groups[j][1])
                    if dim in n_fg:
                        n_fg[dim].update(set(following_groups[j][dim]))
                # Delete groups based on the found overlaps
                for j, dim in reversed(overlaps):
                    del following_groups[j][dim]
                    if len(following_groups[j]) == 0:
                        del output_groups[j]
                        del following_groups[j]
                        len_deleted += 1
                        if j < idx:
                            idx -= 1
                ln -= len_deleted
                output_groups[idx] = n_og
                following_groups[idx] = n_fg
            idx += 1

        # Discard dimension information from following groups since it is not needed anymore
        del_idx = []
        for idx, fs in enumerate(following_groups):
            merged_fg = set()
            merged_og = set()
            for fgi in fs.values():
                merged_fg.update(set(fgi))
            merged_og.update(set(output_groups[idx][1]))
            if len(merged_fg) == 0:
                del_idx.append(idx)
            else:
                following_groups[idx] = merged_fg
                output_groups[idx] = merged_og
        for idx in reversed(del_idx):
            del following_groups[idx]
            del output_groups[idx]

        # Trim groups
        zipped = list(zip(output_groups, following_groups))
        zipped.sort(key=lambda x: len(x[0]), reverse=True)
        del_idx = []
        for idx, (og, fg) in enumerate(zipped):
            for j_idx, (j_og, j_fg) in enumerate(zipped[idx + 1 :], start=idx + 1):
                if j_og.issubset(og):
                    del_idx.append(j_idx)
        output_groups, following_groups = (
            [a for a, _ in zipped],
            [b for _, b in zipped],
        )
        for idx in reversed(del_idx):
            del output_groups[idx]
            del following_groups[idx]

        uses_MHA = any(
            [
                isinstance(m, torch.nn.MultiheadAttention)
                for _, m in model.named_modules()
            ]
        )
        if uses_MHA:
            """
            Since the pytorch implementation of MHA does not allow for independent embedding and qkv sizes,
            we need to merge the MHA proj groups into the main embedding group.
            """
            MHA_groups_idx = []
            for idx, og in enumerate(output_groups):
                for param in og:
                    if "attn.in_proj_weight" in param:
                        module_prefix = param.replace(".in_proj_weight", "")
                        for j, j_og in enumerate(output_groups):
                            if any([module_prefix in p for p in j_og]):
                                if j != idx:
                                    MHA_groups_idx.append((j, idx))
            del_idx = []
            for dst, src in MHA_groups_idx:
                output_groups[dst].update(output_groups[src])
                following_groups[dst].update(following_groups[src])
                del_idx.append(src)
            for idx in reversed(sorted(del_idx)):
                del output_groups[idx]
                del following_groups[idx]

        return output_groups, following_groups

    def trace_dim_stop_and_coupling_ops(
        self, op: int, dim=(1,), print=lambda *args: 0, depth=-1
    ):
        """
        Accumulates all stop and coupling ops along the output nodes of the given op.

        Furthermore, it also calculates the dimension along which the ops channel dimension is
        encountered at the stop or coupling op.

        To print debug output, set print variable to the global print function.
        """
        op_name = self.get_node_name(op)
        if len(dim) == 0:
            if op_name == "UnsqueezeBackward0":
                unsq_dim = self.node_attr[op]["dim"]
                dim = (unsq_dim,)
            else:
                return [], []
        print(op_name, dim)
        s_ops, c_ops = [], []

        ######################
        if op_name == "ExpandBackward0":
            """
            TBackward is short for transpose if only two dimensions are present.
            """
            in_shape, out_shape = (
                self.node_attr[op]["input_shapes"],
                self.node_attr[op]["output_shapes"],
            )
            print(op_name, dim, in_shape, out_shape)

        ######################
        if op_name == "TBackward0":
            """
            TBackward is short for transpose if only two dimensions are present.
            """
            in_shape, out_shape = (
                self.node_attr[op]["input_shapes"],
                self.node_attr[op]["output_shapes"],
            )
            new_dim = list(dim)
            if len(new_dim) == 2:
                new_dim = [new_dim[1], new_dim[0]]
            print(op_name, dim, new_dim, in_shape, out_shape)
            dim = new_dim

        ######################
        if op_name == "TransposeBackward0":
            """
            TransposeBackward0 relies on the "dim0/1" attr of the node.
            dim0 and dim1 contain the channel indexes to be transposed.
            """
            in_shape, out_shape = (
                self.node_attr[op]["input_shapes"],
                self.node_attr[op]["output_shapes"],
            )
            d0, d1 = self.node_attr[op]["dim0"], self.node_attr[op]["dim1"]
            if d0 > len(in_shape[0]):
                d0 = len(in_shape[0]) - 18446744073709551615 + d0 - 1
            if d1 > len(in_shape[0]):
                d1 = len(in_shape[0]) - 18446744073709551615 + d1 - 1
            new_dim = list(dim)  # copy
            for idx, d in enumerate(new_dim):
                if d == d0:
                    new_dim[idx] = d1
                elif d == d1:
                    new_dim[idx] = d0
            print(op_name, dim, new_dim, d0, d1, in_shape, out_shape)
            dim = tuple(new_dim)

        ######################
        if op_name == "PermuteBackward0":
            """
            PermuteBackward0 permutes the channel indexes based on the node "dims" attr

            "dims" contains the applied permutation going from the input to output shape
            """
            permutation = self.node_attr[op]["dims"]
            new_dim = list(dim)  # copy
            for idx, d in enumerate(new_dim):
                for j, p in enumerate(permutation):
                    if p == d:
                        new_dim[idx] = j
            print(op_name, dim, new_dim, permutation)
            dim = tuple(new_dim)

        ######################
        if op_name in [
            "ReshapeAliasBackward0",
            "ViewBackward0",
            "UnsafeViewBackward0",
        ]:
            """
            These ops rely on matching the dimension sizes going from input to output.
            """
            in_shape, out_shape = (
                self.node_attr[op]["input_shapes"],
                self.node_attr[op]["output_shapes"],
            )
            dim_mappings = self.try_match_reshape_dims(in_shape[0], out_shape[0])
            if not all([d in dim_mappings for d in dim]):
                print(op_name, dim, in_shape, out_shape, dim_mappings)
                raise IndexError("traced dim not in dim_mappings")
            new_dim = list(dim)  # copy
            for idx, d in enumerate(new_dim):
                new_dim[idx] = dim_mappings[d]
            new_dim = flatten(new_dim)
            print(op_name, dim, new_dim, in_shape, out_shape)
            dim = tuple(new_dim)

        ######################
        if op_name == "UnsqueezeBackward0":
            """
            Unsqueeze inserts a dimension with size 1 at the given "dim"
            """
            in_shape, out_shape = (
                self.node_attr[op]["input_shapes"],
                self.node_attr[op]["output_shapes"],
            )
            unsq_dim = self.node_attr[op]["dim"]
            if unsq_dim > len(in_shape[0]):
                unsq_dim = len(in_shape[0]) - 18446744073709551615 + unsq_dim - 1
            new_dim = list(dim)  # copy
            for idx, d in enumerate(new_dim):
                if unsq_dim < d:
                    new_dim[idx] += 1
            print(
                op_name,
                dim,
                new_dim,
                unsq_dim,
                self.node_attr[op]["dim"],
                in_shape,
                out_shape,
            )
            dim = tuple(new_dim)

        ######################
        if op_name in ["SumBackward1", "MeanBackward1"]:
            """
            Sum and mean will cause the dimensions in the "dim" attr to be removed.
            """
            in_shape, out_shape = (
                self.node_attr[op]["input_shapes"],
                self.node_attr[op]["output_shapes"],
            )
            unsq_dim = self.node_attr[op]["dim"]
            adjusted_dims = []
            for d in unsq_dim:
                if d > len(in_shape[0]):
                    d = len(in_shape[0]) - 18446744073709551615 + d - 1
                    adjusted_dims.append(d)
                else:
                    adjusted_dims.append(d)
            new_dim = list(dim)  # copy
            for ad_idx in adjusted_dims:
                for idx, d in enumerate(new_dim):
                    if ad_idx == d:
                        del new_dim[idx]
                    elif ad_idx < d:
                        new_dim[idx] -= 1
            print(
                op_name,
                dim,
                new_dim,
                adjusted_dims,
                self.node_attr[op]["dim"],
                in_shape,
                out_shape,
            )
            dim = tuple(new_dim)

        ######################
        if op_name == "BmmBackward0":
            """
            Bmm backward causes the output channel dimensions to always include the last
            two dimension.
            """
            in_shape, out_shape = (
                self.node_attr[op]["input_shapes"],
                self.node_attr[op]["output_shapes"],
            )
            last_dim = len(out_shape[0]) - 1
            new_dim = set(dim)  # copy
            new_dim.add(last_dim)
            new_dim.add(last_dim - 1)
            print(
                op_name,
                dim,
                new_dim,
                in_shape,
                out_shape,
            )
            dim = tuple(new_dim)

        ######################
        if op_name == "SelectBackward0":
            """
            Select takes one entry from the dimension at the given index. This essentiall removes the given dimension
            """
            in_shape, out_shape = (
                self.node_attr[op]["input_shapes"],
                self.node_attr[op]["output_shapes"],
            )
            selected_dim = self.node_attr[op]["dim"]
            new_dim = list(dim)  # copy
            if selected_dim in new_dim:
                new_dim.remove(selected_dim)
            for idx, d in enumerate(new_dim):
                if d > selected_dim:
                    new_dim[idx] -= 1
            print(
                op_name,
                selected_dim,
                dim,
                new_dim,
                in_shape,
                out_shape,
            )
            dim = tuple(new_dim)
        ######################
        if op_name in COUPLING_OPS:
            c_ops.append((op, dim))
        if op_name in STOP_OPS:
            s_ops.append((op, dim))
            if "Norm" not in op_name:
                # Don't stop on Norm, instead just accumulate. Makes handling the pruning easier.
                return s_ops, c_ops
        for outp in self.get_node_out_edges(op):
            s_op, c_op = self.trace_dim_stop_and_coupling_ops(
                outp, dim=dim, print=print, depth=depth
            )
            s_ops.extend(s_op)
            c_ops.extend(c_op)
        return s_ops, c_ops

    def try_match_reshape_dims(self, in_shape: tuple, out_shape: tuple):
        """
        Generates mapping which describes to relation of a channel index in input tensor to output tensor
        based on the input and output shape of a reshape operation
        """

        def match(t1, t2):
            """
            Matches the dimension between the two given shapes by sequentially trying to match the dimension sizes.

            For equal sizes on the same dimension index, a 1:1 mapping is assumed.

            For different sizes on the same dimension index,
            try to accumulate all subsequent indexes until the product of the values
            of the indexes is equal to the value in the first shape at the current index.

            E.g. t1 = (1, 64), t2 = (1, 8, 8) would result in a mapping of {0:[0], 1:[1,2]}

            t1 should be the dimension-wise smaller shape, i.e. len(t1) <= len(t2)
            """
            mappings = defaultdict(list)
            for idx, _ in enumerate(t1):
                not_mapped = []
                for j, v in enumerate(t2):
                    if v == 1:
                        continue
                    if j not in mappings:
                        not_mapped.append(j)
                candidates = []
                for j in not_mapped:
                    if t1[idx] % t2[j] == 0:
                        candidates.append(j)
                        if np.prod([t2[k] for k in candidates]) == t1[idx]:
                            break
                    else:
                        break
                for j in candidates:
                    mappings[j].append(idx)
            return mappings

        if len(in_shape) == len(out_shape):
            # TODO: Handle this better
            mappings = match(out_shape, in_shape)
            alt_mappings = match(in_shape, out_shape)
            if len(mappings) < len(alt_mappings):
                reversed_mappings = defaultdict(list)
                for k, v in alt_mappings.items():
                    if len(v) != 1:
                        raise ValueError("Mapping list length is not one")
                    else:
                        reversed_mappings[v[0]].append(k)
                mappings = reversed_mappings
        elif len(in_shape) > len(out_shape):
            mappings = match(out_shape, in_shape)
        else:
            mappings = match(in_shape, out_shape)
            reversed_mappings = defaultdict(list)
            for k, v in mappings.items():
                if len(v) != 1:
                    raise ValueError("Mapping list length is not one")
                else:
                    reversed_mappings[v[0]].append(k)
            mappings = reversed_mappings

        return mappings
