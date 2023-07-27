
class launchBefore:
    def __init__(self, setup_func, instance_method=False):
        self.setup_func = setup_func
        self.instance_method=instance_method
        
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            if self.instance_method:
                self.setup_func(args[0])
            else:
                self.setup_func()
            return func(*args, **kwargs)
        return wrapper