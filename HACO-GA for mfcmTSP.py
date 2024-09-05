import random
import time
from collections import deque
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm


class HACO_GA:
    def __init__(self, distances, coordinates, num_ants=100, beta=3, gamma=2, population_size=20, generations=500,
                 initial_mutation_rate=0.2, final_mutation_rate=0.02):
        self.distances = np.array(distances)
        self.num_customers = len(distances) - 1  # Exclude depot
        self.num_ants = num_ants
        self.beta = beta
        self.gamma = gamma
        self.population_size = population_size
        self.generations = generations
        self.initial_mutation_rate = initial_mutation_rate
        self.final_mutation_rate = final_mutation_rate
        self.current_mutation_rate = initial_mutation_rate
        self.savings = self.precompute_savings()
        self.best_solution = None
        self.best_cost = float('inf')
        self.coordinates = np.array(coordinates)

    def solve_tsp(self):
        population = self.initialize_population()
        print(f"Initial best cost: {min(individual[1] for individual in population)}")

        for generation in range(self.generations):
            self.update_mutation_rate(generation)

            new_population = []
            new_population.extend(sorted(population, key=lambda x: x[1])[:2])  # Elitism

            while len(new_population) < self.population_size:
                parent1 = self.roulette_wheel_selection(population)
                parent2 = self.roulette_wheel_selection(population)
                child = self.crossover(parent1[0], parent2[0])
                child = self.mutate(child)
                cost = self.calculate_cost(child)
                new_population.append((child, cost))

            population = new_population

            current_best = min(population, key=lambda x: x[1])
            if current_best[1] < self.best_cost:
                self.best_solution = current_best[0]
                self.best_cost = current_best[1]
                print(f"\nGeneration {generation + 1}: New best cost {self.best_cost:.2f}")

        print(f"\nFinal best cost: {self.best_cost:.2f}")
        return self.best_solution, self.best_cost

    # Initialization Methods
    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            solution = self.construct_ant_solution()
            cost = self.calculate_cost(solution)
            population.append((solution, cost))
        return population

    def precompute_savings(self):
        n = self.num_customers + 1
        savings = np.zeros((n, n))
        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    savings[i, j] = (self.distances[i, 0] + self.distances[0, j] - self.distances[i, j]) ** self.gamma
        return savings

    # Solution Construction Methods
    def construct_ant_solution(self):
        unvisited = set(range(1, self.num_customers + 1))
        tour = [0]
        current = 0
        while unvisited:
            probabilities = self.calculate_probabilities(current, unvisited)
            if sum(probabilities) == 0:
                # If all probabilities are zero, choose randomly
                next_customer = random.choice(list(unvisited))
            else:
                next_customer = random.choices(list(unvisited), weights=probabilities, k=1)[0]
            tour.append(next_customer)
            unvisited.remove(next_customer)
            current = next_customer
        tour.append(0)
        return tour

    def calculate_probabilities(self, current, unvisited):
        unvisited_array = np.array(list(unvisited))
        heuristic = (1 / (self.distances[current, unvisited_array] + 1e-10)) ** self.beta
        saving = self.savings[current, unvisited_array]
        probabilities = heuristic * saving

        # Add a small constant to avoid zero probabilities
        probabilities += 1e-10

        # Normalize probabilities
        probabilities /= probabilities.sum()

        return probabilities

    # Genetic Algorithm Components
    def roulette_wheel_selection(self, population):
        total_fitness = sum(1 / individual[1] for individual in population)
        pick = random.uniform(0, total_fitness)
        current = 0
        for individual in population:
            current += 1 / individual[1]
            if current > pick:
                return individual
        return population[-1]

    def crossover(self, parent1, parent2):
        parent1 = np.array(parent1)
        parent2 = np.array(parent2)
        size = len(parent1)
        start, end = sorted(np.random.choice(range(1, size - 1), 2, replace=False))
        child = np.full(size, -1)
        child[start:end] = parent1[start:end]
        remaining = parent2[~np.isin(parent2, child[start:end])]
        child[:start] = remaining[:start]
        child[end:] = remaining[start:]
        return child.tolist()

    def mutate(self, solution):
        solution = np.array(solution)
        if np.random.random() < self.current_mutation_rate:
            idx1, idx2 = np.random.choice(range(1, len(solution) - 1), 2, replace=False)
            solution[idx1], solution[idx2] = solution[idx2], solution[idx1]
        return solution.tolist()

    def update_mutation_rate(self, current_generation):
        progress = current_generation / self.generations
        self.current_mutation_rate = self.initial_mutation_rate - (
                self.initial_mutation_rate - self.final_mutation_rate) * progress

    # Cost Calculation Methods
    def calculate_cost(self, solution):
        return np.sum(self.distances[solution[:-1], solution[1:]])


class MFCMTSP:
    def __init__(self, num_customers=50, truck_speed=1, motorbike_speed=1, drone_speed=1, motorbike_capacity=3):
        self.num_customers = num_customers
        self.area_size = 1500  # 1500x1500 m² area
        self.depot = np.array([self.area_size / 2, self.area_size / 2])  # Depot at center
        self.customers = self.generate_customers()
        self.vehicles = self.initialize_vehicles(truck_speed, motorbike_speed, drone_speed, motorbike_capacity)
        self.distances = self.calculate_distances()
        self.drone_accessible = np.random.choice([True, False], size=num_customers, p=[0.8, 0.2])
        self.graphs = self.create_graphs()

    def initialize_vehicles(self, truck_speed, motorbike_speed, drone_speed, motorbike_capacity):
        return [
            {"type": "truck", "speed": truck_speed, "capacity": float('inf')},
            {"type": "motorbike", "speed": motorbike_speed, "capacity": motorbike_capacity},
            {"type": "drone", "speed": drone_speed, "capacity": 1}
        ]

    def generate_customers(self):
        return np.random.rand(self.num_customers, 2) * self.area_size

    def calculate_distances(self):
        all_points = np.vstack((self.depot, self.customers))
        return squareform(pdist(all_points))

    def create_graphs(self):
        graphs = []
        for vehicle in self.vehicles:
            G = nx.Graph()
            G.add_node(0)  # Depot
            for i in range(1, self.num_customers + 1):
                G.add_node(i)
                if vehicle['type'] == 'truck':
                    # Truck can access all nodes
                    G.add_edge(0, i, weight=self.distances[0][i])
                    for j in range(1, i):
                        G.add_edge(i, j, weight=self.distances[i][j])
                elif vehicle['type'] == 'motorbike':
                    # Motorbike can't access some edges (simplified: can't access 20% of edges)
                    if np.random.rand() > 0.2:
                        G.add_edge(0, i, weight=self.distances[0][i])
                    for j in range(1, i):
                        if np.random.rand() > 0.2:
                            G.add_edge(i, j, weight=self.distances[i][j])
                else:  # Drone
                    # Drone can only go between depot and customers (star topology)
                    if self.drone_accessible[i - 1]:
                        G.add_edge(0, i, weight=self.distances[0][i])
            graphs.append(G)
        return graphs

    def run_experiment(self):
        sol_T, sol_M, sol_D, makespan = self.solve()
        single_truck_sol, single_truck_makespan = self.solve_truck_tsp()

        improvement = (single_truck_makespan - makespan) / single_truck_makespan * 100
        CT = (len(sol_T) - 2) / self.num_customers * 100
        CV = (len(sol_M) - len(sol_M) // 4 * 2) / self.num_customers * 100
        CD = (len(sol_D) - len(sol_D) // 2) / self.num_customers * 100

        cost_T = self.calculate_route_cost(sol_T, 0)
        cost_M = self.calculate_route_cost(sol_M, 1)
        cost_D = self.calculate_route_cost(sol_D, 2)

        return improvement, CT, CV, CD, cost_T, cost_M, cost_D, single_truck_makespan, makespan

    def calculate_route_cost(self, solution, vehicle_idx):
        G = self.graphs[vehicle_idx]
        return sum(G[solution[i]][solution[i + 1]]['weight'] for i in range(len(solution) - 1) if
                   G.has_edge(solution[i], solution[i + 1]))

    def calculate_makespan(self, solution, vehicle_idx):
        G = self.graphs[vehicle_idx]
        makespan = 0
        capacity = self.vehicles[vehicle_idx]['capacity']
        current_load = 0
        for i in range(len(solution) - 1):
            if solution[i] == 0:
                current_load = 0
            else:
                current_load += 1
            if current_load > capacity:
                return float('inf')  # Invalid solution
            if G.has_edge(solution[i], solution[i + 1]):
                makespan += G[solution[i]][solution[i + 1]]['weight'] / self.vehicles[vehicle_idx]['speed']
            else:
                return float('inf')  # Invalid solution
        return makespan

    def is_valid_subset(self, subset, vehicle_idx):
        G = self.graphs[vehicle_idx]
        if vehicle_idx == 2:  # Drone
            return all(self.drone_accessible[node - 1] for node in subset)
        # Check if all edges exist, including connections to depot
        return (G.has_edge(0, subset[0]) and
                G.has_edge(subset[-1], 0) and
                all(G.has_edge(subset[i], subset[i + 1]) for i in range(len(subset) - 1)))

    def calculate_subset_cost(self, subset, vehicle_idx):
        G = self.graphs[vehicle_idx]
        cost = 0
        penalty = 1e6  # Large penalty for non-existent edges

        # Cost from depot to first customer
        cost += G[0][subset[0]]['weight'] if G.has_edge(0, subset[0]) else penalty

        # Cost between customers
        for i in range(len(subset) - 1):
            if G.has_edge(subset[i], subset[i + 1]):
                cost += G[subset[i]][subset[i + 1]]['weight']
            else:
                cost += penalty

        # Cost from last customer back to depot
        cost += G[subset[-1]][0]['weight'] if G.has_edge(subset[-1], 0) else penalty

        return cost / self.vehicles[vehicle_idx]['speed']

    def solve(self):
        print("Solving TSP for truck...")
        sol_T, M_T = self.solve_truck_tsp()
        print(f"Initial truck makespan: {M_T:.2f}")

        sol_M, sol_D = deque(), deque()
        M_M, M_D = 0, 0

        print("Balancing makespans...")

        while M_T > M_M or M_T > M_D:
            diff_M, diff_D = M_T - M_M, M_T - M_D
            if diff_M >= diff_D:
                vehicle_idx, vehicle_type = 1, 'motorbike'
            else:
                vehicle_idx, vehicle_type = 2, 'drone'

            k_V = self.vehicles[vehicle_idx]['capacity']
            min_cost, sol_xy, x, y = float('inf'), [], None, None

            for i in range(1, len(sol_T) - k_V):
                subset = sol_T[i:i + k_V]
                if self.is_valid_subset(subset, vehicle_idx):
                    curr_cost = self.calculate_subset_cost(subset, vehicle_idx)
                    if curr_cost < min_cost:
                        min_cost = curr_cost
                        sol_xy = subset
                        x, y = i, i + k_V - 1

            if x is None or y is None:
                break  # No valid subset found, exit the loop

            if vehicle_type == 'motorbike' and min_cost + M_M < M_T:
                sol_M.extend([0] + sol_xy + [0])
                M_M += min_cost
                removed_cost = self.calculate_subset_cost(sol_T[x - 1:y + 2], 0)
                M_T -= removed_cost
                M_T += self.calculate_subset_cost([sol_T[x - 1], sol_T[y + 1]], 0)
                sol_T = sol_T[:x] + sol_T[y + 1:]
            elif vehicle_type == 'drone' and min_cost + M_D < M_T:
                sol_D.extend([0] + sol_xy + [0])
                M_D += min_cost
                removed_cost = self.calculate_subset_cost(sol_T[x - 1:x + 2], 0)
                M_T -= removed_cost
                M_T += self.calculate_subset_cost([sol_T[x - 1], sol_T[x + 1]], 0)
                sol_T = sol_T[:x] + sol_T[x + 1:]
            else:
                break  # No improvement possible, exit the loop

        print(f"Final makespans - Truck: {M_T:.2f}, Motorbike: {M_M:.2f}, Drone: {M_D:.2f}")
        return list(sol_T), list(sol_M), list(sol_D), max(M_T, M_M, M_D)

    def plot_route(self, solution, color, label, vehicle_idx):
        """
        Plot a route for a specific vehicle.

        Args:
            solution (list): The route to plot.
            color (str): Color to use for plotting the route.
            label (str): Label for the route in the legend.
            vehicle_idx (int): Index of the vehicle in the self.vehicles list.
        """
        G = self.graphs[vehicle_idx]
        for i in range(len(solution) - 1):
            if G.has_edge(solution[i], solution[i + 1]):
                start = self.depot if solution[i] == 0 else self.customers[solution[i] - 1]
                end = self.depot if solution[i + 1] == 0 else self.customers[solution[i + 1] - 1]
                plt.plot([start[0], end[0]], [start[1], end[1]], c=color, linewidth=1, alpha=0.7)

        plt.plot([], [], c=color, label=label)  # Add label to legend

    @staticmethod
    def generate_datasets(start=100, end=1000, step=100):
        return [MFCMTSP(num_customers=n) for n in range(start, end + 1, step)]

    def solve_truck_tsp(self):
        coordinates = [self.depot] + list(self.customers)
        haco_ga = HACO_GA(self.distances, coordinates)
        solution, cost = haco_ga.solve_tsp()
        return solution, cost

# Experiment running functions
def run_speed_experiments(datasets, speed_configs, pbar):
    results = []
    for dataset in datasets:
        for truck_speed, motorbike_speed, drone_speed in speed_configs:
            problem = MFCMTSP(dataset.num_customers, truck_speed, motorbike_speed, drone_speed)
            improvement, CT, CV, CD, cost_T, cost_M, cost_D, initial_makespan, final_makespan = problem.run_experiment()
            results.append((dataset.num_customers, (truck_speed, motorbike_speed, drone_speed), improvement, CT, CV, CD,
                            cost_T, cost_M, cost_D, initial_makespan, final_makespan))
            pbar.update(1)
    return results


def run_capacity_experiments(datasets, capacities, motorbike_speeds, pbar):
    results = []
    for dataset in datasets:
        for capacity in capacities:
            for speed in motorbike_speeds:
                problem = MFCMTSP(dataset.num_customers, motorbike_speed=speed, motorbike_capacity=capacity)
                improvement, CT, CV, CD, cost_T, cost_M, cost_D, initial_makespan, final_makespan = problem.run_experiment()
                results.append((dataset.num_customers, capacity, speed, improvement, CT, CV, CD, cost_T, cost_M, cost_D,
                                initial_makespan, final_makespan))
                pbar.update(1)
    return results


def plot_fleet_utilization_and_improvement(results, speed_config, figure_number):
    customer_counts = sorted(set(result[0] for result in results))
    CT = [next(r[3] for r in results if r[0] == c and r[1] == speed_config) for c in customer_counts]
    CV = [next(r[4] for r in results if r[0] == c and r[1] == speed_config) for c in customer_counts]
    CD = [next(r[5] for r in results if r[0] == c and r[1] == speed_config) for c in customer_counts]
    impr = [next(r[2] for r in results if r[0] == c and r[1] == speed_config) for c in customer_counts]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    x = range(len(customer_counts))

    # Stacked bar chart
    ax1.bar(x, CT, label='CT', color='orange')
    ax1.bar(x, CV, bottom=CT, label='CV', color='gray')
    ax1.bar(x, CD, bottom=[i + j for i, j in zip(CT, CV)], label='CD', color='yellow')

    ax1.set_xlabel('Number of customers')
    ax1.set_ylabel('Fleet utilization & makespan improv (%)')
    ax1.set_ylim(0, 100)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(c) for c in customer_counts])

    # Improvement line
    ax2 = ax1.twinx()
    ax2.plot(x, impr, color='black', marker='*', label='impr.')
    ax2.set_ylabel('Improvement (%)')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title(f'Figure {figure_number}: s_T = {speed_config[0]}, s_M = {speed_config[1]}, s_D = {speed_config[2]}')
    plt.tight_layout()
    plt.show()


def plot_improvement_vs_capacity(capacity_results, speed_config, figure_number):
    capacities = sorted(set(result[1] for result in capacity_results))
    customer_counts = [100, 500, 1000]

    plt.figure(figsize=(10, 6))

    for c in customer_counts:
        improvements = [next((r[3] for r in capacity_results if r[0] == c and r[1] == k), None) for k in capacities]
        plt.plot(capacities, improvements, marker='o', label=f'C={c // 100}00')

    plt.xlabel('k')
    plt.ylabel('Percentage of improvement (%)')
    plt.title(f'Figure {figure_number}: s_T = {speed_config[0]}, s_M = {speed_config[1]}, s_D = {speed_config[2]}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def print_results(speed_results, capacity_results, total_time):
    print("Speed Experiment Results:")
    print("=" * 80)
    for customers in sorted(set(r[0] for r in speed_results)):
        print(f"\nResults for {customers} customers:")
        print("-" * 60)
        for result in [r for r in speed_results if r[0] == customers]:
            speeds = result[1]
            print(f"Speeds (T, M, D): {speeds}")
            print(f"  Initial makespan: {result[7]:.2f}")
            print(f"  Final best makespan: {result[8]:.2f}")
            print(f"  Improvement: {result[2]:.2f}%")
            print(
                f"  Customer Distribution - Truck: {result[3]:.2f}%, Motorbike: {result[4]:.2f}%, Drone: {result[5]:.2f}%")
            print(f"  Route Costs - Truck: {result[6]:.2f}, Motorbike: {result[7]:.2f}, Drone: {result[8]:.2f}")
            print()

    print("\nCapacity Experiment Results:")
    print("=" * 80)
    for customers in sorted(set(r[0] for r in capacity_results)):
        print(f"\nResults for {customers} customers:")
        print("-" * 60)
        for capacity in sorted(set(r[1] for r in capacity_results)):
            for speed in sorted(set(r[2] for r in capacity_results)):
                result = next((r for r in capacity_results if r[0] == customers and r[1] == capacity and r[2] == speed),
                              None)
                if result:
                    print(f"Capacity: {capacity}, Motorbike Speed: {speed}x")
                    print(f"  Initial makespan: {result[7]:.2f}")
                    print(f"  Final best makespan: {result[8]:.2f}")
                    print(f"  Improvement: {result[3]:.2f}%")
                    print(
                        f"  Customer Distribution - Truck: {result[4]:.2f}%, Motorbike: {result[5]:.2f}%, Drone: {result[6]:.2f}%")
                    print(f"  Route Costs - Truck: {result[7]:.2f}, Motorbike: {result[8]:.2f}, Drone: {result[9]:.2f}")
                    print()
                else:
                    print(f"No data for Capacity: {capacity}, Speed: {speed}x")

    print(f"\nTotal execution time: {total_time:.2f} seconds")


if __name__ == "__main__":
    start_time = time.time()
    print("Generating datasets...")
    datasets = MFCMTSP.generate_datasets()
    print(f"Generated {len(datasets)} datasets")

    speed_configs = [
        (1, 1, 1), (1, 2, 2), (1, 2, 4),
        (1, 4, 4), (1, 4, 8), (1, 8, 8)
    ]
    capacities = range(2, 11)
    motorbike_speeds = [1, 2, 4]

    # Calculate total number of experiments
    total_experiments = (len(datasets) * len(speed_configs)) + (len(datasets) * len(capacities) * len(motorbike_speeds))

    with tqdm(total=total_experiments, desc="Overall Progress") as pbar:
        print("\nRunning speed experiments...")
        speed_results = run_speed_experiments(datasets, speed_configs, pbar)

        print("\nRunning capacity experiments...")
        capacity_results = run_capacity_experiments(datasets, capacities, motorbike_speeds, pbar)

    end_time = time.time()
    total_time = end_time - start_time

    print_results(speed_results, capacity_results, total_time)

    # Generate Figure 5-10 (fleet utilization and improvement)
    figure_5_10_configs = [(1, 1, 1), (1, 2, 2), (1, 2, 4), (1, 4, 4), (1, 4, 8), (1, 8, 8)]
    for i, config in enumerate(figure_5_10_configs, start=5):
        plot_fleet_utilization_and_improvement(speed_results, config, figure_number=i)

    # Generate Figure 11-13 (improvement vs capacity)
    figure_11_13_configs = [(1, 1, 1), (1, 2, 1), (1, 4, 1)]
    for i, config in enumerate(figure_11_13_configs, start=11):
        capacity_results_filtered = [r for r in capacity_results if
                                     r[2] == config[1]]  # Filter for correct motorbike speed
        plot_improvement_vs_capacity(capacity_results_filtered, config, figure_number=i)
