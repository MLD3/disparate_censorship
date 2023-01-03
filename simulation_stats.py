def get_test_rate(results, suffix):
    _, _, A, T, *_ = results
    return {
        f"Pr[T=1]_{suffix}": T.mean(),
        f"Pr[T=1|A=0]_{suffix}": T[A == 0].mean(),
        f"Pr[T=1|A=1]_{suffix}": T[A == 1].mean(),
    }


def get_hit_rate(results, suffix):
    _, _, A, T, _, Y_obs = results
    return {
        f"Pr[Y_obs=1|T=1]_{suffix}": Y_obs[T == 1].mean(),
        f"Pr[Y_obs=1|T=1,A=0]_{suffix}": Y_obs[(T == 1) & (A == 0)].mean(),
        f"Pr[Y_obs=1|T=1,A=1]_{suffix}": Y_obs[(T == 1) & (A == 1)].mean(),
    }


def get_prevalence(results, suffix):
    _, _, A, _, Y, _ = results
    return {
        f"Pr[Y=1]_{suffix}": Y.mean(),
        f"Pr[Y=1|A=0]_{suffix}": Y[A == 0].mean(),
        f"Pr[Y=1|A=1]_{suffix}": Y[A == 1].mean(),
    }


def get_observed_prevalence(results, suffix):
    _, _, A, _, _, Y_obs = results
    return {
        f"Pr[Y_obs=1]_{suffix}": Y_obs.mean(),
        f"Pr[Y_obs=1|A=0]_{suffix}": Y_obs[A == 0].mean(),
        f"Pr[Y_obs=1|A=1]_{suffix}": Y_obs[A == 1].mean(),
    }


def get_noise_rate(results, suffix):
    _, _, A, _, Y, Y_obs = results
    N = Y_obs != Y
    return {
        f"Pr[Y_obs!=Y]_{suffix}": N.mean(),
        f"Pr[Y_obs!=Y|A=0]_{suffix}": N[A == 0].mean(),
        f"Pr[Y_obs!=Y|A=1]_{suffix}": N[A == 1].mean(),
    }


def get_covariate_rates(results, suffix):
    _, X_enc, A, *_ = results
    return {
        f"Pr[X_enc=1]_{suffix}": X_enc.mean(axis=0),
        f"Pr[X_enc=1|A=0]_{suffix}": X_enc[A == 0].mean(axis=0),
        f"Pr[X_enc=1|A=1]_{suffix}": X_enc[A == 1].mean(axis=0),
    }


def get_all_stats(*args):
    return {
        **get_test_rate(*args),
        **get_hit_rate(*args),
        **get_prevalence(*args),
        **get_observed_prevalence(*args),
        **get_noise_rate(*args),
        **get_covariate_rates(*args),
    }
