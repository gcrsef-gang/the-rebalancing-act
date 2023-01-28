from scipy.special import rel_entr


def jenson_shannon_divergence(distribution1, distribution2):
    average = [(distribution1[i] + distribution2[i])/2 for i in range(distribution1)]
