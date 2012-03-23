import numpy as np
from pathplanning import Path, mutate_path

class Sampler(object):
    def __init__(self, start_path, cost_cb):
        self.start_path = start_path
        self.cost_cb = cost_cb

        self.current = (start_path.copy(), cost_cb(start_path))
        self.best = tuple(self.current)
        self.it_count = 0

        self.accepts = 0
        self.samples = 0

    def sample(self):
        # mutate_path path
        new_path, log_forward, log_inv = mutate_path(self.current[0])
        new_cost = self.cost_cb(new_path)

        log_alpha = log_inv - log_forward

        l = 1.0
        log_p_new = np.log(l) - l * new_cost
        log_p_old = np.log(l) - l * self.current[1]
        log_alpha += log_p_new - log_p_old

        self.samples += 1
        alpha = np.exp(log_alpha)
        if np.random.uniform() < alpha:
            self.accepts += 1
            self.current = (new_path, new_cost)

        if self.current[1] < self.best[1]:
            self.best = self.current
        
        return self.current

