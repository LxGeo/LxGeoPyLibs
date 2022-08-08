from collections import OrderedDict

class FixSizeOrderedDict(OrderedDict):
    def __init__(self, *args, max=0, on_del_lambda=None, before_delete_check=None, **kwargs):
        self._max = max
        self.on_del_lambda=on_del_lambda if on_del_lambda else lambda x:None
        self.before_delete_check=before_delete_check if before_delete_check else lambda x:True
        super().__init__(*args, **kwargs)
    
    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        if self._max > 0:
            if len(self) > self._max:
                self.popitem(False)                
    
    def __delitem__(self, key):
        deleted_item = (key, self[key])
        assert self.before_delete_check(deleted_item)
        self.on_del_lambda(deleted_item)        
        OrderedDict.__delitem__(self, key)