from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional
from dataclasses import dataclass
import numpy as np


class IndividualBase(ABC):
    @abstractmethod
    def get_values(self):
        pass


class Individual:
    def __init__(self, params: Dict, value: List[np.array] = None):
        if value is None:
            self.value = [
                np.random.uniform(params['lower_bound'],
                                  params['upper_bound'], 1)[0]
                for _ in range(params['number_of_genes'])
            ]
        else:
            self.value = value

    def get_values(self) -> List[np.array]:
        return self.value


class PopulationBase(ABC):

    @abstractmethod
    def get_parents(self):
        pass

    @abstractmethod
    def crossover(self):
        pass

    @abstractmethod
    def mutate(self):
        pass




class Population(PopulationBase):
    _SIZE_KEY = "size"
    _N_PARENTS_KEY = "n_parents"
    _OFFSPRING_SIZE_KEY = "offspring_size"
    _LOWER_BOUND_KEY = "lower_bound"
    _UPPER_BOUND_KEY = "upper_bound"
    _MUTATION_MEAN_KEY = "mutation_mean"
    _MUTATION_SD_KEY = "mutation_sd"

    def __init__(self,
                 pop_parameters: Dict,
                 ind_parameters: Dict,
                 fitness_function: Callable,
                 ind_values: Optional[List[np.array]] = None):
        if ind_values is None:
            self.individuals = [
                Individual(ind_parameters)
                for _ in range(pop_parameters[self._SIZE_KEY])
            ]
        else:
            self.individuals = [
                Individual(ind_parameters, value) for value in ind_values
            ]
        self.fitness_function = fitness_function
        self.parents = None
        self.size = pop_parameters[self._SIZE_KEY]
        self.n_parents = pop_parameters[self._N_PARENTS_KEY]
        self.offspring_size = pop_parameters[self._OFFSPRING_SIZE_KEY]

    def get_parents(self, n_parents: int) -> np.array:
        parents = [self.individuals[i].get_values() for i in range(self.size)]
        parents.sort(key=lambda x: self.fitness_function(*x), reverse=True)
        return np.array(parents[:n_parents])

    def crossover(self, parents: np.array) -> np.array:
        offspring = np.empty(self.offspring_size)
        crossover_point = np.uint8(self.offspring_size[0] / 2)
        for k in range(self.offspring_size[0]):
            parent1_idx = k % parents.shape[0]

            # Index of the second parent to mate
            parent2_idx = (k + 1) % parents.shape[0]
            # Half of the first parent
            offspring[k, 0:crossover_point] = parents[parent1_idx,
                                                      0:crossover_point]
            # Half if the second parent
            offspring[k, crossover_point:] = parents[parent2_idx,
                                                     crossover_point:]
        return offspring

    def mutate(self, offspring_crossover: np.array, ind_params: Dict, pop_params: Dict) -> np.array:
        for idx in range(offspring_crossover.shape[0]):
            # select randomly the gene where randomness is going to be added
            g = np.random.choice(range(offspring_crossover.shape[1]))

            # The random value to be added to the gene.
            offspring_crossover[idx][g] = offspring_crossover[idx][g] + \
                np.random.normal(
                    pop_params[self._MUTATION_MEAN_KEY], pop_params[self._MUTATION_SD_KEY], 1)

        # Apply upper and lower bounds
        offspring_crossover = np.where(
            offspring_crossover > ind_params[self._UPPER_BOUND_KEY],
            ind_params[self._UPPER_BOUND_KEY],
            offspring_crossover
        )
        offspring_crossover = np.where(
            offspring_crossover < ind_params[self._LOWER_BOUND_KEY],
            ind_params[self._LOWER_BOUND_KEY],
            offspring_crossover
        )

        return offspring_crossover
    

@dataclass
class Solution:
    best_individual: np.array
    best_score: Optional[float]


class EvolutionBase(ABC):
    @abstractmethod
    def step(self):
        pass


class Evolution:
    _N_PARENTS_KEY = "n_parents"
    _SIZE_KEY = "size"

    def __init__(self, pop_parameters: Dict,
                 ind_parameters: Dict,
                 fitness_function: Callable
                 ):
        self.fitness = fitness_function
        self.population = Population(
            pop_parameters,
            ind_parameters,
            fitness_function
        )
        self.ind_parameters = ind_parameters
        self.pop_parameters = pop_parameters
        self._best_individual = None
        self._best_score = None

    def step(self) -> None:
        parents = self.population.get_parents(
            self.pop_parameters[self._N_PARENTS_KEY])
        offspring = self.population.crossover(parents)
        mutation = self.population.mutate(
            offspring, self.ind_parameters, self.pop_parameters)
        population_values = [
            self.population.individuals[i].get_values()
            for i in range(self.population.size)
        ]
        population_values.extend(mutation)
        population_values.sort(key=lambda x: self.fitness(*x), reverse=True)
        population_values = population_values[:
                                              self.pop_parameters[self._SIZE_KEY]]
        self._best_individual = population_values[:1]
        self._best_score = self.fitness(*self._best_individual[0])

        # update population
        self.population = Population(
            self.pop_parameters,
            self.ind_parameters,
            self.fitness,
            population_values
        )

    @property
    def solution(self) -> Optional[Solution]:
        if self._best_individual is not None:
            return Solution(self._best_individual[0], self._best_score)


