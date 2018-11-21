from OptFun import*

dEdx_min = dtransdx(psi_x, soilM_plot, psi_sat, b, psi_63, w_exp, Kmax)
dlamdt_min = lam_plot * dEdx_min * alpha / lai
lam_min = np.cumsum(np.diff(res.x) * dlamdt_min[1:])
plt.plot(res.x[1:], lam_plot[0]+lam_min)