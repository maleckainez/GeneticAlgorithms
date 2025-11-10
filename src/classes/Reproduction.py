import numpy as np

from src.classes.ChildrenHandler import ChildrenHandler
from src.classes.ExperimentConfig import ExperimentConfig
from src.classes.PathResolver import PathResolver
from src.classes.PopulationHandler import PopulationHandler


class Reproduction:

    def __init__(self,
            parent_pool: np.ndarray,
            config: ExperimentConfig,
            paths: PathResolver,
            pop_manager: PopulationHandler,
    ):
        self.parent_pool = parent_pool
        self.config = config
        self.paths = paths
        self.pop_manager = pop_manager
        self.parent_pairs: np.ndarray
        self.rng = self.config.rng
        self._pair_parents()

    def single_crossover(self):
        self._calculation_runner(self._kernel_single)

    def double_crossover(self):
        self._calculation_runner(self._kernel_double)

    def _pair_parents(self):
        self.parent_pairs = self.rng.permutation(self.parent_pool).reshape(-1, 2)

    def _setup(self):
        self.stream_batch = self.config.stream_batch_size
        self.crossover_probability = self.config.crossover_probability
        self.mutation_probability = self.config.mutation_probability
        self.population = self.pop_manager.get_pop_handle()
        self.children_manager = ChildrenHandler(
            config=self.config, paths=self.paths, genome_length=self.population.shape[1]
        )
        self.children = self.children_manager.get_children_handle()

    def _kernel_single(self, c1, c2, p1, p2, mask, parent_indices):
        crossover_points = self.rng.integers(
            1, self.children.shape[1], size=len(parent_indices)
        )
        for i in range(len(parent_indices)):
            if mask[i]:
                cut_point = crossover_points[i]
                c1[i, cut_point:] = p2[i, cut_point:]
                c2[i, cut_point:] = p1[i, cut_point:]
        return c1, c2

    def _kernel_double(self, c1, c2, p1, p2, mask, parent_indices):
        first_crossover_points = self.rng.integers(
            1, c1.shape[1] - 1, size=len(parent_indices)
        )
        second_crossover_points = self.rng.integers(
            first_crossover_points + 1, c1.shape[1], size=len(parent_indices)
        )
        crossover_points = np.column_stack(
            (first_crossover_points, second_crossover_points)
        )
        for i in range(len(parent_indices)):
            if mask[i]:
                first_cut_point, second_cut_point = crossover_points[i]
                c1[i, first_cut_point:] = p2[i, first_cut_point:]
                c1[i, second_cut_point:] = p1[i, second_cut_point:]
                c2[i, first_cut_point:] = p1[i, first_cut_point:]
                c2[i, second_cut_point:] = p2[i, second_cut_point:]
        return c1, c2

    def _calculation_runner(self, kernel):
        self._setup()
        for start in range (0, len(self.parent_pairs), self.stream_batch):
            stop = min(start + self.stream_batch, len(self.parent_pairs))
            parent_indices = self.parent_pairs[start:stop]
            p1 = self.population[parent_indices[:,0]]
            p2 = self.population[parent_indices[:,1]]
            c1, c2 = p1.copy(), p2.copy()
            mask = (
                self.rng.random(size=stop - start) < self.crossover_probability
            )
            c1, c2 = kernel(c1, c2, p1, p2, mask, parent_indices)
            if self.mutation_probability > 0:
                self._mutation(c1, c2)
            self.children[start * 2 : stop * 2] = np.concatenate(
                (c1, c2), axis=0
            )
            self.children.flush()
        self.children_manager.close()

    def _mutation(self, c1, c2):
            mask1 = (self.rng.random(size=c1.shape) < self.mutation_probability)
            mask2 = (self.rng.random(size=c2.shape) < self.mutation_probability)
            c1[mask1] ^= 1
            c2[mask2] ^= 1
            return c1, c2

