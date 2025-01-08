from scipy.stats import linregress


def tac_slope(tac_list_current, index, test_range):
    previous_tac_list = tac_list_current[index - test_range:index]
    x = [i for i in range(0, len(previous_tac_list))]
    slope, intercept, r, p, se = linregress(x, previous_tac_list)
    return slope
