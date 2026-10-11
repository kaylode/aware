from typing import List, Dict

class Retriever:
    k: int = 1024
    _is_fitted: bool = False
    return_distances: bool = False
    fields_to_use: List[str] = None
    def fit(self, items:List[Dict]=None, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    def retrieve(self, queries:List[Dict]=None, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")
