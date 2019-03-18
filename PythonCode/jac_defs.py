import numpy as np

# -------- Jacobian definitions ---------

# --------- No comp - No mesophyll -------

def jac_nocomp_nomeso(t, y, ca, cp_interp, k1_interp, k2_interp, VPDinterp):
    """

    :param t: time
    :param y: y[0] is the lambda value and y[1] is the soil moisture value
    :return: returns the jacobian of the BVP system at different times
    """
    jacjac = np.zeros((y.shape[0], y.shape[0], t.shape[0]))
    a = 1.6
    cp = cp_interp(t)
    k1 = k1_interp(t)
    k2 = k2_interp(t)
    VPD = VPDinterp(t)

    numerator = a * (ca - cp) ** 2 * k1 * (cp + k2) ** 2 * VPD * (ca + k2 - 2 * a * VPD * y[0]) ** 3
    denominator = 2 * (a * (ca - cp) * (cp + k2) * VPD * y[0] * (ca + k2 - 2 * a * VPD * y[0]) ** 2 *
                       (ca + k2 - a * VPD * y[0])) ** (3/2)

    jacjac[1, 0, :] = - numerator / denominator
    return jacjac