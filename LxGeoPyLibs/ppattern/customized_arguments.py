from functools import wraps

def customize_arg(arg_name, preprocessing, required=False):
    def inner_decorator(func):
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            if arg_name not in kwargs:
                if required:
                    raise Exception(f"Missing required argument '{arg_name}'")
                else: 
                    return func(*args, **kwargs)
            
            kwargs[arg_name] = preprocessing(kwargs[arg_name])  # preprocess the argument
            return func(*args, **kwargs)
            
        return wrapper
    return inner_decorator