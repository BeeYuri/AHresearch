import logging
logging.basicConfig(level=logging.DEBUG)

class log:
    def __init__(self):
        self
    def struct_log(self):
        #创建logger，如果参数为空则返回root logger
        self.logger = logging.getLogger("log")
        self.logger.setLevel(logging.DEBUG)  #设置logger日志等级
        #创建handler
        self.fh = logging.FileHandler("AHlog.log",encoding="utf-8")
        #设置输出日志格式
        self.formatter = logging.Formatter(
            fmt="%(asctime)s %(name)s %(filename)s %(message)s",
            datefmt="%Y/%m/%d %X"
            )
        self.fh.setFormatter(self.formatter)
        self.logger.addHandler(self.fh)