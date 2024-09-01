import math
import random
import time
import matplotlib.pyplot as plt
import numpy as np


class VRP:
    def __init__(self, filename):
        self.load_problem(filename)

    def load_problem(self, filename):
        with open(filename, 'r') as file:
            lines = file.readlines()

        self.capacity = int([line for line in lines if line.startswith('CAPACITY')][0].split()[-1])
        self.coordinates = []
        self.demands = []

        # Strip whitespace from lines
        lines = [line.strip() for line in lines]

        # Extract the optimal value and number of vehicles
        comment_line = [line for line in lines if line.startswith('COMMENT')][0]

        # Extract optimal value
        optimal_value_part = comment_line.split('Optimal value:')[-1].split(')')[0].strip()
        self.optimal_value = int(optimal_value_part)

        # Extract number of vehicles
        if 'Min no of trucks:' in comment_line:
            vehicles_part = comment_line.split('Min no of trucks:')[-1].split(',')[0].strip()
        else:
            vehicles_part = comment_line.split('No of trucks:')[-1].split(',')[0].strip()
        self.num_vehicles = int(vehicles_part)

        try:
            coord_start = lines.index('NODE_COORD_SECTION') + 1
        except ValueError:
            raise ValueError("NODE_COORD_SECTION not found in the file")

        try:
            demand_start = lines.index('DEMAND_SECTION') + 1
        except ValueError:
            raise ValueError("DEMAND_SECTION not found in the file")

        try:
            depot_start = lines.index('DEPOT_SECTION') + 1
        except ValueError:
            raise ValueError("DEPOT_SECTION not found in the file")

        for line in lines[coord_start:demand_start - 1]:
            _, x, y = map(int, line.split())
            self.coordinates.append((x, y))

        for line in lines[demand_start:depot_start - 1]:
            _, demand = map(int, line.split())
            self.demands.append(demand)

        self.distances = self.calculate_distances()
        self.print_loaded_data()

    def print_loaded_data(self):
        print("Capacity:", self.capacity)
        print("Optimal Value:", self.optimal_value)
        print("Coordinates:", self.coordinates)
        print("Demands:", self.demands)
        print("Distances:")
        for i in range(len(self.distances)):
            print(f"From {i}: {self.distances[i]}")

    def calculate_distances(self):
        num_customers = len(self.coordinates)
        distances = np.zeros((num_customers, num_customers))
        for i in range(num_customers):
            for j in range(num_customers):
                if i != j:
                    dist = math.sqrt((self.coordinates[i][0] - self.coordinates[j][0]) ** 2 +
                                     (self.coordinates[i][1] - self.coordinates[j][1]) ** 2)
                    distances[i][j] = round(dist)  # Round to the nearest integer
        return distances


class HACO_VRP:
    """
        Hybrid Ant Colony Optimization for Vehicle Routing Problem (HACO-VRP)

        This class implements a hybrid algorithm combining Ant Colony Optimization (ACO)
        and Genetic Algorithm (GA) techniques to solve the Vehicle Routing Problem (VRP).
    """
    def __init__(self, vrp, num_ants=100, beta=3, gamma=2, max_iterations=100,
                 population_size=20, generations=200, mutation_prob=0.2):
        self.vrp = vrp
        self.num_ants = num_ants
        self.beta = beta
        self.gamma = gamma
        self.max_iterations = max_iterations
        self.num_customers = len(vrp.coordinates)
        self.num_vehicles = vrp.num_vehicles  # Use the number of vehicles from VRP
        self.max_vehicles = vrp.num_vehicles
        self.best_cost = float('inf')
        self.best_solution = None
        self.best_known_solution = vrp.optimal_value  # Use the optimal value from the dataset
        self.all_solutions = []
        self.best_solutions = []
        self.population_size = population_size
        self.generations = generations
        self.mutation_prob = mutation_prob
        self.rd1 = None
        self.rd2 = None
        self.current_positions = np.zeros(self.num_ants, dtype=int)
        self.tabu_list = np.zeros((self.num_ants, self.num_customers), dtype=np.int32)
        self.tabu_mask = np.ones((self.num_ants, self.num_customers), dtype=bool)
        self.distance_matrix = np.array(self.vrp.distances) # Convert pre-calculated distances to numpy array
        self.savings = self.precompute_savings()


    # Main Algorithm
    def run(self):
        population = self.initialize_population(self.population_size)
        for generation in range(self.generations):
            new_population = []

            # Elitism: carry over the best individuals
            population = sorted(population, key=lambda x: x[1])
            new_population.extend(population[:2])

            while len(new_population) < self.population_size:
                parent1 = self.roulette_wheel_selection(population)
                parent2 = self.roulette_wheel_selection(population)

                if parent1 is None or parent2 is None:
                    print("Error: Selected parent is None")
                    continue

                # Pass the full solutions to crossover, not flattened ones
                child1, child2 = self.partially_matched_crossover(parent1[0], parent2[0])

                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                child1 = [self.two_opt(route) for route in child1 if route]
                child2 = [self.two_opt(route) for route in child2 if route]

                child1_cost = self.calculate_cost(child1)
                child2_cost = self.calculate_cost(child2)

                new_population.append((child1, child1_cost))
                new_population.append((child2, child2_cost))

            population = sorted(new_population, key=lambda x: x[1])[:self.population_size]
            best_solution = population[0]

            if best_solution[1] < self.best_cost:
                self.best_cost = best_solution[1]
                self.best_solution = best_solution[0]

            self.all_solutions.append(best_solution[1])
            print(f"Generation {generation + 1}, Best cost so far: {self.best_cost}")

        self.rd1 = self.calculate_rd1()
        self.rd2 = self.calculate_rd2()

        print(f"Final best solution: {self.best_solution}, Best cost: {self.best_cost}")
        print(f"RD1 (Best known solution deviation): {self.rd1}")
        print(f"RD2 (Solution stability): {self.rd2}")
        return self.best_solution, self.best_cost

    # Initialization Methods
    def initialize_population(self, population_size):
        """
                Initializes the population with random solutions.

                Args:
                    population_size (int): The number of solutions to generate in the population.

                Returns:
                    list: A list of tuples, each containing a solution (list of routes) and its cost (float).
        """
        population = []
        attempts = 0
        max_attempts = population_size * 10  # Limit the number of attempts to avoid infinite loop
        while len(population) < population_size and attempts < max_attempts:
            solution, cost = self.construct_solution()
            if cost != float('inf'):
                population.append((solution, cost))
            attempts += 1
        if not population:
            raise RuntimeError("Failed to initialize a valid population.")
        return population

    def initialize_ants(self):
        """
               Initializes the ants for a new solution construction phase.

               This method resets the ants' positions, tabu lists, and available capacities.
        """
        self.current_positions = np.random.choice(range(1, self.num_customers), self.num_ants, replace=True)
        self.tabu_list.fill(0)
        self.tabu_mask.fill(True)
        for i, pos in enumerate(self.current_positions):
            self.update_tabu(i, pos, 1, 1)
        self.available_capacity = np.full(self.num_ants, self.vrp.capacity)

    # Genetic Algorithm Components
    def roulette_wheel_selection(self, population):
        """
               Selects a solution from the population using roulette wheel selection.

               Args:
                   population (list): A list of tuples, each containing a solution and its cost.

               Returns:
                   tuple: The selected solution (list of routes) and its cost (float).
        """
        max_cost = max(individual[1] for individual in population)
        total_fitness = sum(max_cost - individual[1] for individual in population)
        pick = random.uniform(0, total_fitness)
        current = 0
        for individual in population:
            current += max_cost - individual[1]
            if current > pick:
                return individual
        return population[-1]  # Return the last individual as a fallback

    def partially_matched_crossover(self, parent1, parent2):
        child1 = [[] for _ in range(self.max_vehicles)]
        child2 = [[] for _ in range(self.max_vehicles)]

        # Flatten the parents
        parent1_flat = [c for route in parent1 for c in route if c != 0]
        parent2_flat = [c for route in parent2 for c in route if c != 0]

        # Ensure all customers are included
        all_customers = set(range(1, self.num_customers))
        parent1_flat.extend(all_customers - set(parent1_flat))
        parent2_flat.extend(all_customers - set(parent2_flat))

        size = len(parent1_flat)

        # Perform PMX on flattened parents
        start, end = sorted(random.sample(range(size), 2))

        # Create the crossover mapping
        mapping1 = dict(zip(parent1_flat[start:end], parent2_flat[start:end]))
        mapping2 = dict(zip(parent2_flat[start:end], parent1_flat[start:end]))

        # Apply crossover
        child1_flat = parent1_flat[:]
        child2_flat = parent2_flat[:]

        for i in range(size):
            if i < start or i >= end:
                # Apply mapping to child1
                while child1_flat[i] in mapping1:
                    child1_flat[i] = mapping1[child1_flat[i]]
                # Apply mapping to child2
                while child2_flat[i] in mapping2:
                    child2_flat[i] = mapping2[child2_flat[i]]

        # Convert flattened children back to route format
        for child_flat, child in [(child1_flat, child1), (child2_flat, child2)]:
            vehicle = 0
            capacity = self.vrp.capacity
            for customer in child_flat:
                if vehicle >= self.max_vehicles:
                    # If we've used all vehicles, add remaining customers to the last route
                    child[-1].insert(-1, customer)
                elif capacity >= self.vrp.demands[customer]:
                    if len(child[vehicle]) == 0:
                        child[vehicle].append(0)
                    child[vehicle].append(customer)
                    capacity -= self.vrp.demands[customer]
                else:
                    if len(child[vehicle]) > 0:
                        child[vehicle].append(0)
                    vehicle += 1
                    if vehicle < self.max_vehicles:
                        child[vehicle] = [0, customer]
                        capacity = self.vrp.capacity - self.vrp.demands[customer]

            # Complete any unfinished routes
            for route in child:
                if len(route) > 0 and route[-1] != 0:
                    route.append(0)

            # Remove empty routes
            child[:] = [route for route in child if len(route) > 2]

        return child1, child2

    def mutate(self, solution):
        if random.random() < self.mutation_prob:
            solution = self.swap_mutation(solution)
        if random.random() < self.mutation_prob:
            solution = self.inversion_mutation(solution)
        return solution

    def swap_mutation(self, solution):
        flat_solution = [c for route in solution for c in route if c != 0]
        if len(flat_solution) > 1:
            idx1, idx2 = random.sample(range(len(flat_solution)), 2)
            flat_solution[idx1], flat_solution[idx2] = flat_solution[idx2], flat_solution[idx1]
        return self.unflatten_routes(flat_solution)

    def inversion_mutation(self, solution):
        flat_solution = [c for route in solution for c in route if c != 0]
        if len(flat_solution) > 1:
            idx1, idx2 = sorted(random.sample(range(len(flat_solution)), 2))
            flat_solution[idx1:idx2] = flat_solution[idx1:idx2][::-1]
        return self.unflatten_routes(flat_solution)

    # Ant Colony Optimization Components
    def construct_solution(self):
        self.initialize_ants()
        best_solution = None
        best_cost = float('inf')

        for ant in range(self.num_ants):
            ant_solution = []
            unvisited = set(range(1, self.num_customers))

            while unvisited and len(ant_solution) < self.max_vehicles:
                route = [0]
                capacity_left = self.vrp.capacity

                while unvisited:
                    current = route[-1]
                    next_customer = self.select_next_customer(ant, current, capacity_left, unvisited)

                    if next_customer is None or self.vrp.demands[next_customer] > capacity_left:
                        break

                    route.append(next_customer)
                    unvisited.remove(next_customer)
                    capacity_left -= self.vrp.demands[next_customer]

                route.append(0)
                ant_solution.append(route)

            # If there are still unvisited customers, create additional routes
            while unvisited:
                if len(ant_solution) >= self.max_vehicles:
                    # If we've used all vehicles, add remaining customers to the last route
                    for customer in unvisited:
                        ant_solution[-1].insert(-1, customer)
                    break
                else:
                    route = [0]
                    capacity_left = self.vrp.capacity
                    while unvisited and capacity_left >= min(self.vrp.demands[c] for c in unvisited):
                        next_customer = min(unvisited, key=lambda c: self.vrp.demands[c])
                        if self.vrp.demands[next_customer] <= capacity_left:
                            route.append(next_customer)
                            unvisited.remove(next_customer)
                            capacity_left -= self.vrp.demands[next_customer]
                        else:
                            break
                    route.append(0)
                    ant_solution.append(route)

            cost = self.calculate_cost(ant_solution)
            if cost < best_cost:
                best_solution = ant_solution
                best_cost = cost

        return best_solution, best_cost

    def construct_route(self, visited):
        """
        Constructs a single route for an ant.

        Args:
            visited (set): A set of customers already visited.

        Returns:
            list: A single route constructed by the ant.
        """
        route = [0]
        capacity_left = self.vrp.capacity
        unvisited = set(range(1, self.num_customers)) - visited

        while unvisited:
            current = route[-1]
            next_customer = self.select_next_customer(0, current, capacity_left,
                                                      unvisited)  # Use 0 as a placeholder for ant_index
            if next_customer is None:
                break
            next_customer = int(next_customer)
            capacity_left -= self.vrp.demands[next_customer]
            route.append(next_customer)
            visited.add(next_customer)
            unvisited.remove(next_customer)

        route.append(0)
        return route

    def select_next_customer(self, ant_index, current, capacity_left, unvisited):
        feasible = [j for j in unvisited if self.vrp.demands[j] <= capacity_left]

        if not feasible:
            return None

        probabilities = []
        for j in feasible:
            heuristic = (1 / (self.distance_matrix[current][j] + 1e-10)) ** self.beta
            saving = self.savings[current, j]  # Use precomputed savings
            probability = heuristic * saving
            probabilities.append((j, probability))

        if not probabilities:
            return None

        total = sum(p[1] for p in probabilities)

        if total == 0:
            # If all probabilities are zero, choose randomly
            return random.choice(feasible)

        probabilities = [(p[0], p[1] / total) for p in probabilities]

        r = random.random()
        cumulative_probability = 0
        for j, prob in probabilities:
            cumulative_probability += prob
            if r <= cumulative_probability:
                return j

        return probabilities[-1][0]  # Fallback to last customer if rounding errors occur

    def update_tabu(self, ant_index, customer, vehicle_number, sequence):
        """
                Updates the tabu list for a specific ant after visiting a customer.

                Args:
                    ant_index (int): The index of the ant.
                    customer (int): The customer that was visited.
                    vehicle_number (int): The current vehicle number.
                    sequence (int): The sequence number of the visit within the current route.
        """
        self.tabu_list[ant_index, customer] = (vehicle_number << 16) | sequence
        self.tabu_mask[ant_index, customer] = False

    # Local Search Methods
    def two_opt(self, route):
        """
                       Applies the 2-opt local search improvement to a single route.

                       Args:
                           route (list): A list representing a single route.

                       Returns:
                           list: The improved route after applying 2-opt.
        """
        improvement = True
        best_distance = self.calculate_route_cost(route)
        while improvement:
            improvement = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route)):
                    if j - i == 1:
                        continue
                    new_route = route[:i] + route[i:j][::-1] + route[j:]
                    new_distance = self.calculate_route_cost(new_route)
                    if new_distance < best_distance:
                        route = new_route
                        best_distance = new_distance
                        improvement = True
                        break
                if improvement:
                    break
        return route

    # Solution Handling Methods
    def flatten_routes(self, solution):
        """
               Flattens a solution (list of routes) into a single list.

               Args:
                   solution (list): A list of routes.

               Returns:
                   list: A flattened representation of the solution.
        """
        return [customer for route in solution for customer in route if customer != 0]

    def unflatten_routes(self, flat_solution):
        routes = []
        current_route = [0]
        capacity_left = self.vrp.capacity

        for customer in flat_solution:
            if self.vrp.demands[customer] <= capacity_left:
                current_route.append(customer)
                capacity_left -= self.vrp.demands[customer]
            else:
                current_route.append(0)
                routes.append(current_route)
                current_route = [0, customer]
                capacity_left = self.vrp.capacity - self.vrp.demands[customer]

        if len(current_route) > 1:
            current_route.append(0)
            routes.append(current_route)

        # If we have more routes than allowed vehicles, merge the last routes
        while len(routes) > self.max_vehicles:
            last_route = routes.pop()
            routes[-1] = routes[-1][:-1] + last_route[1:]

        return routes

    # Cost Calculation Methods
    def calculate_cost(self, solution):
        total_cost = 0
        visited_customers = set()

        for route in solution:
            route_cost = 0
            route_load = 0
            for i in range(len(route) - 1):
                route_cost += self.distance_matrix[route[i]][route[i + 1]]
                if route[i] != 0:
                    visited_customers.add(route[i])
                    route_load += self.vrp.demands[route[i]]
            total_cost += route_cost

            # Penalize capacity violations
            if route_load > self.vrp.capacity:
                total_cost += 1000000 * (route_load - self.vrp.capacity)

        # Penalize solutions that don't visit all customers
        missing_customers = self.num_customers - 1 - len(visited_customers)
        if missing_customers > 0:
            total_cost += 1000000 * missing_customers

        # Penalize solutions with too many vehicles
        if len(solution) > self.max_vehicles:
            total_cost += 1000000 * (len(solution) - self.max_vehicles)

        return total_cost

    def calculate_route_cost(self, route):
        cost = 0
        for i in range(len(route) - 1):
            cost += self.distance_matrix[route[i]][route[i + 1]]
        return cost

    def calculate_rd1(self):
        """
                Calculates the relative percentage deviation from the best known solution.

                Returns:
                    float: The percentage deviation from the best known solution.
        """
        return (self.best_cost - self.best_known_solution) / self.best_known_solution * 100

    def calculate_rd2(self):
        """
               Calculates the relative percentage deviation between all solutions and the best solution.

               This metric provides information about the stability of the algorithm.

               Returns:
                   float: The percentage deviation between all solutions and the best solution.
        """
        mean_all_solutions = np.mean(self.all_solutions)
        return (mean_all_solutions - self.best_cost) / self.best_cost * 100

    # Utility Methods
    def is_customer_visited(self, ant_index, customer):
        """
                Checks if a customer has been visited by a specific ant.

                Args:
                    ant_index (int): The index of the ant.
                    customer (int): The customer to check.

                Returns:
                    bool: True if the customer has been visited, False otherwise.
        """
        return not self.tabu_mask[ant_index, customer]

    def get_unvisited_customers(self, ant_index):
        """
                Gets the list of unvisited customers for a specific ant.

                Args:
                    ant_index (int): The index of the ant.

                Returns:
                    numpy.array: An array of indices of unvisited customers.
        """
        return np.where(self.tabu_mask[ant_index, 1:])[0] + 1

    def precompute_savings(self):
        savings = np.zeros((self.num_customers, self.num_customers))
        for i in range(self.num_customers):
            for j in range(self.num_customers):
                if i != j:
                    savings[i, j] = (self.distance_matrix[i, 0] + self.distance_matrix[0, j] - self.distance_matrix[
                        i, j]) ** self.gamma
        return savings

    # Visualization Methods
    def plot_solution(self, solution, show_plot=True):
        """
               Plots the solution on a 2D graph.

               This method visualizes the routes of the solution, marking the depot and customers.

               Args:
                   solution (list): A list of routes representing the solution to plot.
                   show_plot (bool): If True, displays the plot. Default is True.
        """
        if not show_plot:
            return  # Skip plotting if show_plot is False

        colors = [
            'b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'brown', 'pink', 'olive',
            'cyan', 'teal', 'navy', 'magenta', 'gold', 'lime', 'coral', 'indigo', 'tan',
            'salmon', 'plum', 'turquoise', 'lavender'
        ]

        # Plot all customers with their numbers
        all_x = [self.vrp.coordinates[i][0] for i in range(self.num_customers)]
        all_y = [self.vrp.coordinates[i][1] for i in range(self.num_customers)]
        plt.scatter(all_x, all_y, c='grey', label='All Customers')

        for i in range(1, self.num_customers):
            plt.text(self.vrp.coordinates[i][0], self.vrp.coordinates[i][1], str(i), fontsize=12, ha='right')

        # Plot each route
        for i, route in enumerate(solution):
            x = [self.vrp.coordinates[customer][0] for customer in route]
            y = [self.vrp.coordinates[customer][1] for customer in route]
            plt.plot(x, y, marker='o', color=colors[i % len(colors)], linestyle='-', label=f'Route {i + 1}')

        depot_x, depot_y = self.vrp.coordinates[0]
        plt.plot(depot_x, depot_y, marker='s', color='r', label='Depot')

        # Add the best solution cost to the plot
        best_cost = self.calculate_cost(solution)
        best_cost_text = f'Best cost: {best_cost}'
        plt.text(0.05, 0.95, best_cost_text, transform=plt.gca().transAxes, fontsize=14,
                 verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

        plt.title('HACO-CVRP Solution')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.grid(True)
        plt.show()


def validate_solution(self, solution):
    total_cost = 0
    visited_customers = set()
    for route in solution:
        route_cost = 0
        route_load = 0
        for i in range(len(route) - 1):
            route_cost += self.distance_matrix[route[i]][route[i + 1]]
            if route[i] != 0:  # not depot
                route_load += self.vrp.demands[route[i]]
                visited_customers.add(route[i])
        total_cost += route_cost
        if route_load > self.vrp.capacity:
            print(f"Capacity exceeded in route {route}")
            return False

    if len(visited_customers) != self.num_customers - 1:
        print(f"Not all customers visited. Visited: {len(visited_customers)}, Expected: {self.num_customers - 1}")
        return False

    if len(solution) > self.max_vehicles:
        print(f"Too many vehicles used. Used: {len(solution)}, Maximum allowed: {self.max_vehicles}")
        return False

    if abs(total_cost - self.calculate_cost(solution)) > 1e-6:
        print(f"Cost mismatch. Calculated: {total_cost}, Reported: {self.calculate_cost(solution)}")
        return False

    return True


start_time_total = time.time()
num_runs = 10
best_overall_solution = None
best_overall_cost = float('inf')
best_overall_run_time = 0

all_best_solutions = []
all_best_costs = []

filename = '../Thesis/test/P-n22-k8.vrp'
vrp = VRP(filename)

for run in range(num_runs):
    print(f"Run {run + 1}/{num_runs}")
    start_time_run = time.time()

    aco_sa = HACO_VRP(vrp, num_ants=100, beta=3, gamma=2,
                      population_size=50, generations=1000, mutation_prob=0.2)
    best_solution, best_cost = aco_sa.run()

    all_best_solutions.append(best_solution)
    all_best_costs.append(best_cost)

    end_time_run = time.time()
    run_time = end_time_run - start_time_run

    print(f"Best cost for run {run + 1}: {best_cost}")
    print(f"Time taken for run {run + 1}: {run_time:.2f} seconds")

    if best_cost < best_overall_cost:
        best_overall_solution = best_solution
        best_overall_cost = best_cost
        best_overall_run_time = run_time

total_time = time.time() - start_time_total

average_best_cost = np.mean(all_best_costs)

print(f"\nBest overall cost: {best_overall_cost:.2f}")
print(f"Best overall solution: {best_overall_solution}")
print(f"Run time for best solution: {best_overall_run_time:.2f} seconds")
print(f"Average best cost: {average_best_cost:.2f}")
print(f"Total time taken: {total_time:.2f} seconds")
print(f"Average time per run: {total_time / num_runs:.2f} seconds")

aco_sa.plot_solution(best_overall_solution, show_plot=True)

if validate_solution(aco_sa, best_overall_solution):
    print("Solution is valid.")
else:
    print("Solution is invalid.")
