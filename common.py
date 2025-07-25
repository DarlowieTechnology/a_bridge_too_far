class ConfigSingleton(object):
    """represents TOML configuration file"""
    conf : dict[str, str] = {}
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ConfigSingleton, cls).__new__(cls)
        return cls.instance 

#
# open text file
# return tuple [bool, content]
# if error, return [False, None]
#
def openTextFile(filePath : str, readContent : bool) -> tuple[bool, str] :
    file_path = Path(filePath)
    if not file_path.is_file():
        print(f"***ERROR: Error opening file {filePath}")
        return False, None
    try:
        with open(filePath, "r") as textFile:
            if readContent:
                return True, textFile.read()
            else:
               return True, None
    except FileNotFoundError as e:
        print(f"***ERROR: Error opening file {filePath}, exception {e}")
    except PermissionError as e:
        print(f"***ERROR: Permission error opening file {filePath}, exception {e}")
    except Exception as e:
        print(f"***ERROR: General error opening file {filePath}, exception {e}")
    return False, None
