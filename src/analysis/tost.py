from scipy import stats as _stats


def ci_95(lst):
    """95% CI half-width for a list; None values are silently dropped."""
    lst = [x for x in lst if x is not None]
    n = len(lst)
    if n < 2:
        return 0.0
    return float(_stats.t.ppf(0.975, df=n - 1) * _stats.sem(lst))


def tost_paired(a_list, b_list, margin=0.02):
    """Two one-sided t-tests: is b equivalent to a within ±margin?

    Uses only paired indices where both values are not None.
    Returns a dict with keys: equivalent, p_low, p_high, mean_diff,
    ci_low, ci_high, margin, n.
    """
    pairs = [(a, b) for a, b in zip(a_list, b_list) if a is not None and b is not None]
    n = len(pairs)
    _null = {"equivalent": None, "p_low": None, "p_high": None,
             "mean_diff": None, "ci_low": None, "ci_high": None,
             "margin": margin, "n": n}
    if n < 2:
        return _null
    a_arr, b_arr = zip(*pairs)
    diffs = [b - a for a, b in zip(a_arr, b_arr)]
    # H0_low:  mean_diff <= -margin  (reject → mean_diff > -margin)
    _, p_low  = _stats.ttest_1samp(diffs, -margin, alternative="greater")
    # H0_high: mean_diff >= +margin  (reject → mean_diff < +margin)
    _, p_high = _stats.ttest_1samp(diffs,  margin, alternative="less")
    mean_d  = sum(diffs) / n
    ci_half = _stats.t.ppf(0.975, df=n - 1) * _stats.sem(diffs)
    return {
        "equivalent": bool(p_low < 0.05 and p_high < 0.05),
        "p_low":      round(float(p_low),  4),
        "p_high":     round(float(p_high), 4),
        "mean_diff":  round(float(mean_d), 4),
        "ci_low":     round(float(mean_d - ci_half), 4),
        "ci_high":    round(float(mean_d + ci_half), 4),
        "margin":     margin,
        "n":          n,
    }
