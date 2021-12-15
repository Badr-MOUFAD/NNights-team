class Pipeline():
    def __init__(self, name):
        self.name = name
        self.pipeline = []
    def log(self):
        print(self.name)