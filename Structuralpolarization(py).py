
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.stats import beta

np.random.seed(41134)

# Parameter settings
N = 400
K = 4
beta_prob = 0.1
gamma = 10
w = 0.5
R = 10000
act = 3
sumage = 10
neip = 1

# Generate authentic preference (beta) distributions for N agents
initial_choice = np.linspace(beta.ppf(0.5, 1.001, sumage - 1.001),
                             1 - beta.ppf(0.5, 1.001, sumage - 1.001),
                             N)
np.random.shuffle(initial_choice)

age_alpha = np.zeros(N)
age_beta = np.zeros(N)

for t in range(N):
    fun = lambda x: abs(beta.ppf(0.5, x, sumage - x) - initial_choice[t])
    res = minimize_scalar(fun, bounds=(1.001, sumage - 1.001), method='bounded')
    age_alpha[t] = res.x
    age_beta[t] = sumage - res.x

# Generate social network

network = nx.watts_strogatz_graph(N, K*2, beta_prob)

# Initial system settings
rounds = 0
pro_alpha = np.ones(N)
pro_beta = np.ones(N)
choice = initial_choice.copy()
cmin = 0
cmax = 1
cx = np.arange(cmin, cmax + 0.05, 0.05)
all_choice = np.tile(initial_choice, (R, 1)).T
disutility_age = np.zeros((N, R))

# Utility functions
def utility_a(x, alpha1, beta1, gamma):
    I1 = beta.cdf(x, alpha1, beta1)
    return np.exp(abs(gamma * (I1 - 0.5)))

def utility_s(alpha2, beta2, gamma):
    I2 = beta.cdf(0.5, alpha2, beta2)
    return np.exp(gamma * abs(I2 - 0.5))

# Iterative process
while rounds < R:
    rounds += 1
    i = np.random.randint(N)

    # Information sources are social network neighbors
    neighbors_list = list(network.neighbors(i))
    sample_size = int(neip * len(neighbors_list))
    sample_indices = np.random.choice(len(neighbors_list), sample_size, replace=False)
    sample_choice = choice[np.array(neighbors_list)[sample_indices]]

    x = np.arange(0, 1.001, 0.001)

    # First component of disutility: disutility from abandoning authentic preference
    y1 = utility_a(x, age_alpha[i], age_beta[i], gamma)

    # Bayesian learning of social ranks
    comparison = np.sign(np.subtract.outer(x, sample_choice))
    post_alpha_test = pro_alpha[i] + np.count_nonzero(comparison > 0, axis=1)
    post_beta_test = pro_beta[i] + np.count_nonzero(comparison < 0, axis=1)

    mask = post_alpha_test + post_beta_test > 20
    post_alpha_test[mask] = 20 * post_alpha_test[mask] / (post_alpha_test[mask] + post_beta_test[mask])
    post_beta_test[mask] = 20 * post_beta_test[mask] / (post_alpha_test[mask] + post_beta_test[mask])

    # Second component of disutility: disutility from being extreme
    y2 = np.array([utility_s(a, b, gamma) for a, b in zip(post_alpha_test, post_beta_test)])

    # Aggregated disutility value
    y = (1 - w) * y1 + w * y2
    ystar = np.min(y)
    xstar = np.argmin(y)
    choice[i] = x[xstar]
    all_choice[i, rounds:] = choice[i]
    disutility_age[i, rounds:] = ystar
    pro_alpha[i] = post_alpha_test[xstar]
    pro_beta[i] = post_beta_test[xstar]

    # Probability of changing neighbor correlated with skewness of posterior distribution
    cutting_prob = min(1, (act * abs(pro_alpha[i] - pro_beta[i])) / (pro_alpha[i] + pro_beta[i]))
    if np.random.rand() <= cutting_prob:
        distances = np.abs(sample_choice - initial_choice[i])
        cutting_neighbor = neighbors_list[sample_indices[np.argmax(distances)]]
        candidates = np.setdiff1d(np.arange(N), neighbors_list + [i])
        candidate_distances = np.abs(choice[candidates] - choice[i])
        candidates_potential = candidates[candidate_distances < np.max(distances)]
        if len(candidates_potential) > 0 and network.degree[cutting_neighbor] > 1:
            candidatedegree = np.array([network.degree[c] for c in candidates_potential])
            candidate_prob = candidatedegree / np.sum(candidatedegree)
            new_neighbor = np.random.choice(candidates_potential, p=candidate_prob)
        else:
            new_neighbor = cutting_neighbor

        network.remove_edge(i, cutting_neighbor)
        network.add_edge(i, new_neighbor)

    clustering_coeffs = list(nx.clustering(network).values())
    cf_age = np.mean(clustering_coeffs)
    
    # Some measurement of attitude polarization
    uti_age = np.mean(disutility_age[:, rounds - 1])
    var_age = np.var(choice)
    skewed = np.mean(np.abs(pro_alpha - pro_beta) / (pro_alpha + pro_beta))

# Visualization
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(network)
nodes = nx.draw_networkx_nodes(network, pos, node_size=50, node_color=choice, cmap='coolwarm', vmin=0, vmax=1)
edges = nx.draw_networkx_edges(network, pos, alpha=0.5)
plt.colorbar(nodes, label='Choice value (0 to 1)')
plt.title(f'Round = {rounds}')
plt.axis('off')
plt.show()
