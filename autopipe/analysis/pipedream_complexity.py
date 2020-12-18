"""Since PipeDream does exhaustive search,
    for small graphs, no mixed-pipe, 1 hardware level hierarchy
    it may be better to just use it,
    (however it models communication incorrectly)
    complexity is (simplified from PipeDream's paper)
    L*N^3*m^2

    N - graph nodes (operations/layers)
    m - gpus per level
    L - number of levels

"""

def pipedream_extimated_time(N, m, L=1):
    # compute a mult fact from resnet50 Pipedream's largest network, reported 8 seconds
    baseline_complexity = 709789824 # resnet (N=177,m=8, L=2)
    baseline_seconds = 8

    complexity = L * N**3 * m**2
    estimated_time = baseline_seconds * (complexity / baseline_complexity)

    return estimated_time
