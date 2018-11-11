import numpy as np


class Noise(object):

    def __init__(self, delta, sigma, ou_a, ou_mu):
        # Noise parameters
        self.delta = delta
        self.sigma = sigma
        self.ou_a = ou_a
        self.ou_mu = ou_mu
        print(self.__dict__)

    def brownian_motion_log_returns(self):
        sqrt_delta_sigma = np.sqrt(self.delta) * self.sigma
        return np.random.normal(loc=0, scale=sqrt_delta_sigma, size=None)

    def ornstein_uhlenbeck_level(self, prev_ou_level):
        drift = self.ou_a * (self.ou_mu - prev_ou_level) * self.delta
        randomness = self.brownian_motion_log_returns()
        return prev_ou_level + drift + randomness

if __name__ == "__main__":
    from train import load_config
    config = load_config("config_g2g.yaml")
    noise_params = {k: np.array(list(map(float, v.split(","))))
                         for k, v in config["noise_params"].items()}
    noise = Noise(**noise_params)
    ou_lvl = np.zeros(2)
    v = []
    for i in range(1000000):
        # print(ou_lvl)
        ou_lvl = noise.ornstein_uhlenbeck_level(ou_lvl)
        v.append(ou_lvl)
        if (i+1) % 50 == 0:
            ou_lvl = np.zeros(2)

    import matplotlib.pyplot as plt
    l, a = list(zip(*v))
    plt.hist(a, np.linspace(-1.5, 1.5, 300))
    plt.show()
