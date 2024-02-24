import json
from .structure_analyzer import StructureAnalyzer


class PruneCfg:
    def __init__(self, structure=None, file=None) -> None:
        if structure is None and file is None:
            raise ValueError(f"structure and file arguments can not both be none")
        if structure is not None and file is not None:
            raise ValueError(f"structure and file arguments can not both be set")

        if structure is not None:
            self.init_from_structure(structure)
        if file is not None:
            self.init_from_file(file)

    def init_from_structure(self, structure: StructureAnalyzer):
        zipped = zip(structure.output_groups, structure.out_in_groups)
        self.groups = [
            {"out_group": og, "out_in_group": oig, "type": "BasePruner"}
            for og, oig in zipped
        ]

    def init_from_file(self, filename: str):
        with open(filename) as file:
            data = json.load(file)
            self.groups = data["groups"]

    def save_to_file(self, filename: str):
        data = {"groups": self.groups}
        with open(filename, "w") as file:
            json.dump(data, file, indent=2)
