import numpy as np
import pickle
from PIL import Image


class ACO:
    def __init__(self, alpha=1, beta=2, rho=0.1, psi=0.1, ants=10, iterations=5):
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.psi = psi
        self.ants = ants
        self.iterations = iterations
        self.ant_journeys = []

    def define_complex_region(self, image):
        gray = np.mean(image, axis=2)
        threshold = np.mean(gray)
        return gray > threshold

    def initialize_pheromone(self, shape):
        return np.ones(shape)

    def heuristic_value(self, image):
        gray = np.mean(image, axis=2)
        return gray / (np.max(gray) + 1e-6)

    def transition_probability(self, pheromone, heuristic):
        tau_alpha = np.power(pheromone, self.alpha)
        eta_beta = np.power(heuristic, self.beta)
        prob = tau_alpha * eta_beta
        prob /= np.sum(prob)
        return prob

    def update_pheromone(self, pheromone, ant_paths):
        pheromone *= (1 - self.rho)
        for path in ant_paths:
            pheromone[path] += self.psi
        return pheromone

    def select_ant_paths(self, prob):
        flat_prob = prob.flatten()
        indices = np.random.choice(len(flat_prob), size=self.ants, p=flat_prob)
        return [np.unravel_index(i, prob.shape) for i in indices]

    def embed(self, image, message):
        np_img = np.array(image)
        binary_message = ''.join(format(ord(c), '08b') for c in message)
        bit_idx = 0

        complex_region = self.define_complex_region(np_img)
        pheromone = self.initialize_pheromone(complex_region.shape)
        heuristic = self.heuristic_value(np_img)

        for _ in range(self.iterations):
            prob = self.transition_probability(pheromone, heuristic)
            ant_paths = self.select_ant_paths(prob)
            self.ant_journeys.append(ant_paths)
            pheromone = self.update_pheromone(pheromone, ant_paths)

        for (x, y) in self.ant_journeys[-1]:
            if bit_idx >= len(binary_message):
                break
            pixel = np_img[x, y]
            pixel[0] = (pixel[0] & ~1) | int(binary_message[bit_idx])
            np_img[x, y] = pixel
            bit_idx += 1

        return Image.fromarray(np_img), complex_region, pheromone

    def extract(self, image, complex_region, pheromone):
        np_img = np.array(image)
        extracted_bits = []

        prob = self.transition_probability(pheromone, complex_region)
        ant_paths = self.select_ant_paths(prob)

        for (x, y) in ant_paths:
            extracted_bits.append(str(np_img[x, y][0] & 1))

        chars = [chr(int(''.join(extracted_bits[i:i+8]), 2))
                 for i in range(0, len(extracted_bits), 8)]
        return ''.join(chars)

    # ============================================================
    # SAVE MODEL
    # ============================================================
    def save_model(self, complex_region, pheromone, path="aco_model.pkl"):
        model_data = {
            "alpha": self.alpha,
            "beta": self.beta,
            "rho": self.rho,
            "psi": self.psi,
            "ants": self.ants,
            "iterations": self.iterations,
            "complex_region": complex_region,
            "pheromone": pheromone,
            "ant_journeys": self.ant_journeys
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        print(f"[MODEL SAVED] {path}")

    # ============================================================
    # LOAD MODEL
    # ============================================================
    @staticmethod
    def load_model(path="aco_model.pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
