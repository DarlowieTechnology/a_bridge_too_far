class ConfigSingleton(object):
    """represents TOML configuration file"""
    conf : dict[str, str] = {}
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ConfigSingleton, cls).__new__(cls)
        return cls.instance 

