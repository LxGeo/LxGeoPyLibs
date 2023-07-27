
class launchBefore:
    def __init__(self, setup_func):
        self.setup_func = setup_func
        
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            self.setup_func(*args, **kwargs)
            return func(*args, **kwargs)
        return wrapper