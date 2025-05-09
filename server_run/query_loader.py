from beir.datasets.data_loader import GenericDataLoader
from typing import Dict

class QueryOnlyDataLoader(GenericDataLoader):
    def load_only_queries(self, split="test") -> Dict[str, str]:
        self.check(fIn=self.query_file, ext="jsonl")
        if not len(self.queries):
            self._load_queries()
        return self.queries