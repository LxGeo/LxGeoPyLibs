
class MissingFileException(Exception):    
    def __init__(self, file_path):
        super(MissingFileException, self).__init__(f"Missing file at location: {file_path}")
        
class MissingDirectoryException(Exception):    
    def __init__(self, dir_path):
        super(MissingDirectoryException, self).__init__(f"Missing directory at location: {dir_path}")

class SingleFileWithPatternException(Exception):
    def __init__(self, file_pattern, found_count):
        super(SingleFileWithPatternException, self).__init__(f"Expected one file with pattern {file_pattern}. Found multiple N={found_count}!")

