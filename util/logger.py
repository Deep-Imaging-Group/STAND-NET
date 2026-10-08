from collections import defaultdict

class Recoder:
    def __init__(self):
        self.metrics = defaultdict(list)

    def record(self, name, value):
        if isinstance(value, tuple) or isinstance(value, list):
            self.metrics[name].extend(value)
        elif isinstance(value, float):
            self.metrics[name].append(value)
        else:
            raise("Unrecognise type of number.")

    def summary(self):
        kvs = {}
        for key in self.metrics.keys():
            kvs[key] = sum(self.metrics[key]) / len(self.metrics[key])
            del self.metrics[key][:]
            self.metrics[key] = []
        return kvs