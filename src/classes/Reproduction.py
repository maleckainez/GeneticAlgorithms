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
    ):
        self.parent_pool = parent_pool
        self.config = config
        self.paths = paths
        self.parent_pairs: np.ndarray
        self.rng = self.config.rng
        self._pair_parents()
        

    def single_crossover(self, pop_manager:PopulationHandler, children_manager:ChildrenHandler):
        self._calculation_runner(self._kernel_single, pop_manager, children_manager)

    def double_crossover(self, pop_manager:PopulationHandler, children_manager:ChildrenHandler):
        self._calculation_runner(self._kernel_double, pop_manager, children_manager)

    def _pair_parents(self):
        self.parent_pairs = self.rng.permutation(self.parent_pool).reshape(-1, 2)

    def _setup(self):
        self.stream_batch = self.config.stream_batch_size
        self.crossover_probability = self.config.crossover_probability
        self.mutation_probability = self.config.mutation_probability

    def _kernel_single(self, c1, c2, p1, p2, mask):
        batch_size, genome_length = c1.shape
        cut_columns = self.rng.integers(1, genome_length, size=batch_size)
        column_index = np.arange(genome_length)

        cut_mask = column_index[None,:] >= cut_columns[:, None]
        cut_mask &= mask[:,None]
        c1[cut_mask] = p2[cut_mask]
        c2[cut_mask] = p1[cut_mask]
        return c1, c2

    def _kernel_double(self, c1, c2, p1, p2, mask):
        batch_size, genome_length = c1.shape
        start_cut_col = self.rng.integers(1, genome_length-1, size=batch_size)
        stop_cut_col = self.rng.integers(start_cut_col+1, genome_length, size=batch_size)
        column_index = np.arange(genome_length)
        cut_mask = (column_index[None,:] >= start_cut_col[:,None]) & (column_index[None,:] < stop_cut_col[:,None])
        cut_mask &= mask[:,None]
        c1[cut_mask] = p2[cut_mask]
        c2[cut_mask] = p1[cut_mask]
        return c1, c2

    def _calculation_runner(self, kernel,pop_manager: PopulationHandler,children_manager: ChildrenHandler):
        population = pop_manager.get_pop_handle()
        children = children_manager.get_children_handle()
        self._setup()
        for start in range (0, len(self.parent_pairs), self.stream_batch):
            stop = min(start + self.stream_batch, len(self.parent_pairs))
            parent_indices = self.parent_pairs[start:stop]
            p1 = population[parent_indices[:,0]]
            p2 = population[parent_indices[:,1]]
            c1, c2 = p1.copy(), p2.copy()
            mask = (
                self.rng.random(size=stop - start) < self.crossover_probability
            )
            c1, c2 = kernel(c1, c2, p1, p2, mask)
            if self.mutation_probability > 0:
                self._mutation(c1, c2)
            children[start * 2 : stop * 2] = np.concatenate(
                (c1, c2), axis=0
            )
            children.flush()

    def _mutation(self, c1, c2):
        mask1 = (self.rng.random(size=c1.shape) < self.mutation_probability)
        mask2 = (self.rng.random(size=c2.shape) < self.mutation_probability)
        c1[mask1] ^= 1
        c2[mask2] ^= 1
        return c1, c2
