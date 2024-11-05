import numpy as np
from scipy.stats import beta, norm
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

# Set random seed
np.random.seed(1412431)

# Set parameter values
N = 1000 # number of agents in the system, default value is 1000
K = 10 # number of in-group and out-group samples, default value is 10
gamma = 10 # obedience to social norms, see Appendix.
R = 20000 # number of iterations, default value is 20000
w1 = 0.2 # w_AUT, motivation strength of maintaining private attitudes
wgi2 = 0.5 # relative importance of in-group convergence and out-group divergence, default value is .5
w2 = (1 - w1) * wgi2 # w_IN
w3 = (1 - w1) * (1 - wgi2) # w_OUT

# Parameters for the beta distribution describing authentic preferences
sumage1 = 10  # dimension 1
sumage2 = 10  # dimension 2

# To avoid estimated social norm distributions become too sharp or flat, we set up and down thresholds for 
# the sum of two parameters in the beta distributions.
UL = 30
LL = 3

# Utility function
def utilitya(x, alpha1, beta1, gamma):
    return np.exp(np.abs(gamma * (beta.cdf(x, alpha1, beta1) - 0.5)))

# Helper function for minimizing and generating authentic preference distribution
def optimize_beta_params(target, total_sum):
    def objective(x):
        return abs(beta.ppf(0.5, x, total_sum - x) - target)
    result = minimize_scalar(objective, bounds=(1.001, total_sum - 1.001), method='bounded')
    return result.x, total_sum - result.x

# Generate uniformly distributed private attitudes for two independent issues
alpha1, beta1 = 1.001, 10 - 1.001
limit1, limit2 = beta.ppf(0.5, alpha1, beta1), 1 - beta.ppf(0.5, alpha1, beta1)
Z1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], N)
U1 = norm.cdf(Z1)
X1 = limit1 + U1 * (limit2 - limit1)
int1, int2 = X1[:, 0], X1[:, 1]

# Calculate each agent's authentic preference distribution
agealpha1, agebeta1 = zip(*[optimize_beta_params(int1[t], sumage1) for t in range(N)])
agealpha2, agebeta2 = zip(*[optimize_beta_params(int2[t], sumage2) for t in range(N)])
agealpha1, agebeta1, agealpha2, agebeta2 = map(np.array, [agealpha1, agebeta1, agealpha2, agebeta2])

# Initial system setup
choice1, choice2 = int1.copy(), int2.copy()
allchoice1 = np.tile(int1, (R, 1)).T
allchoice2 = np.tile(int2, (R, 1)).T
ystarm = np.zeros((N, R))
m1m = np.zeros((N, R)) + 0.5

# Function to normalize beta parameters
def normalize_params(expalpha, expbeta, UL, LL):
    sumab = expalpha + expbeta
    if sumab >= UL:
        expalpha, expbeta = UL * (expalpha / sumab), UL * (expbeta / sumab)
    elif sumab <= LL:
        expalpha, expbeta = LL * (expalpha / sumab), LL * (expbeta / sumab)
    return max(expalpha, 1.001), max(expbeta, 1.001)

# Iterative process
for round in range(R):
    i = np.random.randint(0, N)

    # Determine group memberships
    leftgroup1, rightgroup1 = np.where(choice1 < 0.5)[0], np.where(choice1 >= 0.5)[0]
    leftgroup2, rightgroup2 = np.where(choice2 < 0.5)[0], np.where(choice2 >= 0.5)[0]
    
    SI = np.arange(N)
    SO = np.arange(N)
    
    # Generate two k-element sets for in-group and out-group members who share same identities on dimension one
    if i in leftgroup1:

        INP1 = np.random.choice(leftgroup1, K, replace=False)
        OUTP1 = np.random.choice(rightgroup1, K, replace=False)
    
    else:

        INP1 = np.random.choice(rightgroup1, K, replace=False)
        OUTP1 = np.random.choice(leftgroup1, K, replace=False)
     
    # Generate two k-element sets for randomly selected in-group and out-group samples
    SI = np.delete(SI, INP1)
    SO = np.delete(SO, OUTP1)
    INRAND = np.random.choice(SI, K, replace=False) 
    OUTRAND = np.random.choice(SO, K, replace=False)
    
    # Randomly select an m1 value, proportion of samples with same D1 identity, from 11 elements (0:.1:1)
    m1v = np.linspace(0, 1, 11)
    m1p = np.random.choice(m1v, 1, replace=False)
    
    # Generate in-group and out-group samples, including m1*k with same D1 identities, and (1-m1)*k randomly selected. 
    samplesized1, samplesizerand = int(np.ceil(m1p * K)), K - int(np.ceil(m1p * K))
    
    ingroupsample = np.unique(np.concatenate([INP1[:samplesized1], INRAND[:samplesizerand]]))
    outgroupsample = np.unique(np.concatenate([OUTP1[:samplesized1], OUTRAND[:samplesizerand]]))

    ingroupsamplechoice2 = choice2[ingroupsample]
    outgroupsamplechoice2 = choice2[outgroupsample]

    # Function to calculate updated choice
    def update_choice(x, agealpha, agebeta, ingroup_choices, outgroup_choices):
        y1 = utilitya(x, agealpha, agebeta, gamma)
        expalpha_in, expbeta_in = normalize_params(((1 - np.mean(ingroup_choices)) * (np.mean(ingroup_choices) ** 2)) / (np.std(ingroup_choices) ** 2) - np.mean(ingroup_choices),
                                                    ((1 - np.mean(ingroup_choices)) / np.mean(ingroup_choices)) * (((1 - np.mean(ingroup_choices)) * (np.mean(ingroup_choices) ** 2)) / (np.std(ingroup_choices) ** 2) - np.mean(ingroup_choices)), UL, LL)
        y2 = utilitya(x, expalpha_in, expbeta_in, gamma)
        y3 = utilitya(x, *normalize_params(((1 - np.mean(outgroup_choices)) * (np.mean(outgroup_choices) ** 2)) / (np.std(outgroup_choices) ** 2) - np.mean(outgroup_choices),
                                           ((1 - np.mean(outgroup_choices)) / np.mean(outgroup_choices)) * (((1 - np.mean(outgroup_choices)) * (np.mean(outgroup_choices) ** 2)) / (np.std(outgroup_choices) ** 2) - np.mean(outgroup_choices)), UL, LL), gamma) if len(outgroup_choices) > 0 else 0
        y = w1 * y1 + w2 * y2 - w3 * y3
        return x[np.argmin(y)], y[np.argmin(y)]

    x = np.linspace(0, 1, 1001)

    # Calculate optimal expressed attitude on dimension two, and the corresponding minimized disutility value 
    updatexy2 = update_choice(x, agealpha2[i], agebeta2[i], ingroupsamplechoice2, outgroupsamplechoice2)
    
    # Learning process, if the randomly selected m1 value gives lower disutility than the last iteration agent i was activated
    # then agent i will use the updated m1 value to draw samples in the following iterations,
    # otherwise, the agent will not change the m1 value. 
    if round == 0:
        ystarm[i, round:] = updatexy2[1]
        m1 = m1p
        m1m[i, round:] = m1p
        choice2[i] = updatexy2[0]
    else:
        if ystarm[i, round - 1] != 0 and updatexy2[1] < ystarm[i, round - 1]:
            ystarm[i, round:] = updatexy2[1]
            m1 = m1p
            m1m[i, round:] = m1p
            choice2[i] = updatexy2[0]
        elif ystarm[i, round - 1] != 0 and updatexy2[1] >= ystarm[i, round - 1]:
            m1 = m1m[i, round - 1]
        elif ystarm[i, round - 1] == 0:
            ystarm[i, round:] = updatexy2[1]
            m1 = m1p
            m1m[i, round:] = m1p
            choice2[i] = updatexy2[0]
            
    # We now return to the attitude expression on dimension one with the  updated m1 value
    
    samplesized1, samplesizerand = int(np.ceil(m1 * K)), K - int(np.ceil(m1 * K))
    
    ingroupsample = np.unique(np.concatenate([INP1[:samplesized1], INRAND[:samplesizerand]]))
    outgroupsample = np.unique(np.concatenate([OUTP1[:samplesized1], OUTRAND[:samplesizerand]]))

    ingroupsamplechoice1 = choice1[ingroupsample]
    outgroupsamplechoice1 = choice1[outgroupsample]
    updatexy1 = update_choice(x, agealpha1[i], agebeta1[i], ingroupsamplechoice1, outgroupsamplechoice1)
    choice1[i] = updatexy1[0]    

# Visualization
range1 = np.arange(0, 1.05, 0.05)
leftgroup1, rightgroup1 = np.where(choice1 <= 0.5)[0], np.where(choice1 > 0.5)[0]
leftgroup2, rightgroup2 = np.where(choice2 <= 0.5)[0], np.where(choice2 > 0.5)[0]
choice1l, choice1r = choice1[leftgroup1], choice1[rightgroup1]
choice2l, choice2r = choice2[leftgroup1], choice2[rightgroup1]

# Figure 1: Histograms of choice1 and choice2
plt.figure(figsize=(10, 6))
for idx, (choices_l, choices_r, title) in enumerate(zip([choice1l, choice2l], [choice1r, choice2r], ["Choice1", "Choice2"])):
    plt.subplot(1, 2, idx + 1)
    yy_l, _ = np.histogram(choices_l, bins=range1)
    yy_r, _ = np.histogram(choices_r, bins=range1)
    yy_all = yy_l.sum() + yy_r.sum()
    plt.bar(range1[:-1], yy_l / yy_all, width=0.05, color='blue', alpha=0.5, label='Left Group')
    plt.bar(range1[:-1], yy_r / yy_all, width=0.05, bottom=yy_l / yy_all, color='red', alpha=0.5, label='Right Group')
    plt.ylim(0, 0.5)
    plt.title(f"Distribution of {title}")
    plt.legend()
plt.show()

# Figure 2: Scatter plot of initial and final choices
plt.figure(figsize=(8, 8))
plt.scatter(int1, int2, s=5, color='gray', label="Initial Choices", alpha=0.6)
for ar in range(len(int1)):
    plt.arrow(int1[ar], int2[ar], choice1[ar] - int1[ar], choice2[ar] - int2[ar], 
              head_width=0.02, head_length=0.02, color='gray', alpha=0.5)
plt.scatter(choice1[leftgroup1], choice2[leftgroup1], s=20, color='blue', label="Left Group")
plt.scatter(choice1[rightgroup1], choice2[rightgroup1], s=20, color='red', label="Right Group")
plt.xlabel('Choice1')
plt.ylabel('Choice2')
plt.title("Choices at the Stable State")
plt.grid(True)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend()
plt.show()