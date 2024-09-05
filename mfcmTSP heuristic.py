import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import networkx as nx
import time

class MFCMTSP:
    def __init__(self, num_customers=50, truck_speed=1, motorbike_speed=1, drone_speed=1, motorbike_capacity=3):
        self.num_customers = num_customers
        self.area_size = 1500  # 1500x1500 m² area
        self.depot = np.array([self.area_size/2, self.area_size/2])  # Depot at center
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
        single_truck_sol = self.nearest_neighbor(0)
        single_truck_makespan = self.calculate_makespan(single_truck_sol, 0)

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
        cost = 0
        for i in range(len(solution) - 1):
            if G.has_edge(solution[i], solution[i + 1]):
                cost += G[solution[i]][solution[i + 1]]['weight']
        return cost

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
            if G.has_edge(solution[i], solution[i+1]):
                makespan += G[solution[i]][solution[i+1]]['weight'] / self.vehicles[vehicle_idx]['speed']
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

    def nearest_neighbor(self, vehicle_idx):
        G = self.graphs[vehicle_idx]
        unvisited = set(G.nodes()) - {0}
        tour = [0]
        while unvisited:
            last = tour[-1]
            next_node = min(unvisited, key=lambda x: G[last][x]['weight'] if G.has_edge(last, x) else float('inf'))
            if G[last][next_node]['weight'] == float('inf'):
                break  # No accessible nodes left
            tour.append(next_node)
            unvisited.remove(next_node)
        tour.append(0)
        return tour

    def solve(self):
        print("Solving TSP for truck...")
        sol_T = self.nearest_neighbor(0)
        M_T = self.calculate_makespan(sol_T, 0)
        print(f"Initial truck makespan: {M_T:.2f}")

        sol_M, sol_D = [], []
        M_M, M_D = 0, 0

        print("Balancing makespans...")

        while M_T > M_M or M_T > M_D:
            diff_M, diff_D = M_T - M_M, M_T - M_D
            if diff_M >= diff_D:
                vehicle_idx, vehicle_type = 1, 'motorbike'
                k_V = self.vehicles[vehicle_idx]['capacity']
            else:
                vehicle_idx, vehicle_type = 2, 'drone'
                k_V = 1  # Drone always has capacity 1 in the paper's algorithm

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
                sol_T = sol_T[:x] + sol_T[y + 1:]
            elif vehicle_type == 'drone' and min_cost + M_D < M_T:
                sol_D.extend([0] + [sol_T[x]] + [0])  # Drone takes only one customer
                M_D += min_cost
                sol_T = sol_T[:x] + sol_T[x + 1:]  # Remove only one customer
            else:
                break  # No improvement possible, exit the loop

            M_T = self.calculate_makespan(sol_T, 0)

        print(f"Final makespans - Truck: {M_T:.2f}, Motorbike: {M_M:.2f}, Drone: {M_D:.2f}")
        return sol_T, sol_M, sol_D, max(M_T, M_M, M_D)

    def plot_route(self, solution, color, label, vehicle_idx):
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

# Experiment running functions
def run_speed_experiments(datasets, speed_configs):
    results = []
    for i, dataset in enumerate(datasets):
        print(f"Processing dataset {i+1}/{len(datasets)} ({dataset.num_customers} customers)")
        for j, (truck_speed, motorbike_speed, drone_speed) in enumerate(speed_configs):
            print(f"  Speed config {j+1}/{len(speed_configs)}: T={truck_speed}, M={motorbike_speed}, D={drone_speed}")
            problem = MFCMTSP(dataset.num_customers, truck_speed, motorbike_speed, drone_speed)
            improvement, CT, CV, CD, cost_T, cost_M, cost_D, initial_makespan, final_makespan = problem.run_experiment()
            results.append((problem.num_customers, (truck_speed, motorbike_speed, drone_speed), improvement, CT, CV, CD, cost_T, cost_M, cost_D, initial_makespan, final_makespan))
    return results

def run_capacity_experiments(datasets, capacities, motorbike_speeds):
    results = []
    for i, dataset in enumerate(datasets):
        print(f"Processing dataset {i+1}/{len(datasets)} ({dataset.num_customers} customers)")
        for j, capacity in enumerate(capacities):
            for k, speed in enumerate(motorbike_speeds):
                print(f"  Capacity {j+1}/{len(capacities)}, Speed {k+1}/{len(motorbike_speeds)}: C={capacity}, S={speed}")
                problem = MFCMTSP(dataset.num_customers, motorbike_speed=speed, motorbike_capacity=capacity)
                improvement, CT, CV, CD, cost_T, cost_M, cost_D, initial_makespan, final_makespan = problem.run_experiment()
                results.append((dataset.num_customers, capacity, speed, improvement, CT, CV, CD, cost_T, cost_M, cost_D, initial_makespan, final_makespan))
    return results

def plot_fleet_utilization_and_improvement(results, speed_config, figure_number):
    customer_counts = sorted(set(result[0] for result in results))
    CT = [next(r[3] for r in results if r[0] == c and r[1] == speed_config) for c in customer_counts]
    CV = [next(r[4] for r in results if r[0] == c and r[1] == speed_config) for c in customer_counts]
    CD = [next(r[5] for r in results if r[0] == c and r[1] == speed_config) for c in customer_counts]
    impr = [next(r[2] for r in results if r[0] == c and r[1] == speed_config) for c in customer_counts]

    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Stacked bar chart
    x = range(len(customer_counts))
    ax1.bar(x, CT, label='CT', color='orange')
    ax1.bar(x, CV, bottom=CT, label='CV', color='gray')
    ax1.bar(x, CD, bottom=[i + j for i, j in zip(CT, CV)], label='CD', color='yellow')

    ax1.set_xlabel('Number of customers')
    ax1.set_ylabel('Fleet utilization & makespan improv (%)')
    ax1.set_ylim(0, 100)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{c}" for c in customer_counts])

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
    customer_counts = sorted(set(result[0] for result in capacity_results))

    plt.figure(figsize=(12, 8))

    for c in customer_counts:
        improvements = [next((r[3] for r in capacity_results if r[0] == c and r[1] == k and r[2] == speed_config[1]), None) for k in capacities]
        plt.plot(capacities, improvements, marker='o', label=f'C={c}')

    plt.xlabel('Motorbike Capacity (k)')
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
            print(f"  Initial makespan: {result[9]:.2f}")
            print(f"  Final best makespan: {result[10]:.2f}")
            print(f"  Improvement: {result[2]:.2f}%")
            print(f"  Customer Distribution - Truck: {result[3]:.2f}%, Motorbike: {result[4]:.2f}%, Drone: {result[5]:.2f}%")
            print(f"  Route Costs - Truck: {result[6]:.2f}, Motorbike: {result[7]:.2f}, Drone: {result[8]:.2f}")
            print()

    print("\nCapacity Experiment Results:")
    print("=" * 80)
    for customers in sorted(set(r[0] for r in capacity_results)):
        print(f"\nResults for {customers} customers:")
        print("-" * 60)
        for capacity in sorted(set(r[1] for r in capacity_results)):
            for speed in sorted(set(r[2] for r in capacity_results)):
                result = next((r for r in capacity_results if r[0] == customers and r[1] == capacity and r[2] == speed), None)
                if result:
                    print(f"Capacity: {capacity}, Motorbike Speed: {speed}x")
                    print(f"  Initial makespan: {result[10]:.2f}")
                    print(f"  Final best makespan: {result[11]:.2f}")
                    print(f"  Improvement: {result[3]:.2f}%")
                    print(f"  Customer Distribution - Truck: {result[4]:.2f}%, Motorbike: {result[5]:.2f}%, Drone: {result[6]:.2f}%")
                    print(f"  Route Costs - Truck: {result[7]:.2f}, Motorbike: {result[8]:.2f}, Drone: {result[9]:.2f}")
                    print()
                else:
                    print(f"No data for Capacity: {capacity}, Speed: {speed}x")

    print(f"\nTotal execution time: {total_time:.2f} seconds")


if __name__ == "__main__":
    start_time = time.time()
    print("Generating datasets...")
    datasets = MFCMTSP.generate_datasets(100, 1000, 100)
    print(f"Generated {len(datasets)} datasets")

    speed_configs = [
        (1, 1, 1),
        (1, 2, 2),
        (1, 2, 4),
        (1, 4, 4),
        (1, 4, 8),
        (1, 8, 8)
    ]

    capacities = range(2, 11)
    motorbike_speeds = [1, 2, 4]

    print("\nRunning speed experiments...")
    speed_results = run_speed_experiments(datasets, speed_configs)

    print("\nRunning capacity experiments...")
    capacity_results = run_capacity_experiments(datasets, capacities, motorbike_speeds)

    end_time = time.time()
    total_time = end_time - start_time

    print_results(speed_results, capacity_results, total_time)

    print("\nGenerating plots...")
    # Generate Figure 5-10 (fleet utilization and improvement)
    figure_5_10_configs = [(1, 1, 1), (1, 2, 2), (1, 2, 4), (1, 4, 4), (1, 4, 8), (1, 8, 8)]
    for i, config in enumerate(figure_5_10_configs, start=5):
        print(f"Generating Figure {i}...")
        plot_fleet_utilization_and_improvement(speed_results, config, figure_number=i)

    # Generate Figure 11-13 (improvement vs capacity)
    figure_11_13_configs = [(1, 1, 1), (1, 2, 1), (1, 4, 1)]
    for i, config in enumerate(figure_11_13_configs, start=11):
        print(f"Generating Figure {i}...")
        capacity_results_filtered = [r for r in capacity_results if
                                     r[2] == config[1]]  # Filter for correct motorbike speed
        plot_improvement_vs_capacity(capacity_results_filtered, config, figure_number=i)

    print(f"\nTotal execution time: {total_time:.2f} seconds")

