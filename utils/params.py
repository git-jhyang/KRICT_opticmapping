import torch, os, json

class Parameters:
    def __init__(self, fn=None, root='./params_dbpn', default='defaults.json'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._load(os.path.join(root, default))
        if fn is not None:
            self._load(os.path.join(root, fn))
        
    def _load(self, fn):
        with open(fn) as f:
            d = json.load(f)
        for k,v in d.items():
            setattr(self, k, v)
