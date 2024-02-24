from .graph import Graph

import torch


class StructureAnalyzer:
    def __init__(self, model: torch.nn.Module) -> None:
        pass

    def generate_structure(self, model: torch.nn.Module, inputs: torch.Tensor):
        tree = Graph()
        visited = set()

        param_names = {id(v): k for k, v in model.named_parameters()}

        def get_var_name(var):
            return param_names[id(var)] if id(var) in param_names else ""

        SAVED_PREFIX = "_saved_"

        def traverse(fn):
            if fn in visited:
                return
            visited.add(fn)
            if hasattr(fn, "variable"):
                # if grad_accumulator, add the node for `.variable`
                var = fn.variable
                visited.add(var)
                tree.add_node(var, get_var_name(var))
                tree.add_edge(var, fn)

            tree.add_node(fn, str(type(fn).__name__))

            def add_metadata_hook(x, y):
                attrs = dict()
                for attr in dir(fn):
                    if not attr.startswith(SAVED_PREFIX):
                        continue
                    val = getattr(fn, attr)
                    attr = attr[len(SAVED_PREFIX) :]
                    if torch.is_tensor(val):
                        continue
                    elif isinstance(val, tuple) and any(
                        torch.is_tensor(t) for t in val
                    ):
                        continue
                    else:
                        attrs[attr] = val

                inp_shapes, out_shapes = None, None
                if len(x) > 0:
                    inp_shapes = [a.shape for a in x if a is not None]
                if len(y) > 0:
                    out_shapes = [a.shape for a in y if a is not None]
                attrs["input_shapes"] = inp_shapes
                attrs["output_shapes"] = out_shapes
                tree.add_attr_to_node(id(fn), attrs)

            fn.register_hook(add_metadata_hook)

            if hasattr(fn, "next_functions"):
                for u in fn.next_functions:
                    if u[0] is not None:
                        # TODO: u[1] is index, useful for split/concat
                        tree.add_edge(u[0], fn)
                        traverse(u[0])

        out = model(inputs)
        if isinstance(out, tuple):
            out = out[0][-1]
        out.sum().backward(retain_graph=True)
        root = out.grad_fn
        traverse(root)
        out.sum().backward(retain_graph=True)

        og, fg = tree.get_pruning_groups(param_names, model=model)
        self.output_groups = [list(g) for g in og]
        self.out_in_groups = [list(g) for g in fg]
        self.tree = tree
        return
