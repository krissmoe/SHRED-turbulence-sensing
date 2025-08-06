import matplotlib.pyplot as plt
import numpy as np
import utilities3 as utilities
import cmocean
import processdata3 as processdata
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



#TODO: fix file name dependency
def plot_svd_and_spectra(u_total, v_total, s_total, mode_list, psd_multi, k_bins, labels = None, nx=256, ny=256, nt=12500, rank=100, name="surf"):
    """
    Visualize spatial/temporal SVD modes together with singular values and
    power–spectral‐density (PSD) curves.

    A grid of subplots is created:

    * **Rows 0–1:** spatial modes (chosen by ``mode_list``)  
    * **Rows 2–3:** corresponding temporal coefficients  
    * **Bottom row:** (i) normalized singular values and their cumulative
      sum, (ii) PSD spectra for several SVD truncation ranks

    The figure is saved as ``svd_spectra_<name>.pdf`` in the working
    directory and displayed on screen.

    Parameters
    ----------
    u_total : ndarray, shape (nx*ny, 2*rank) or (nx*ny, rank)
        Matrix of left singular vectors (reshaped into spatial modes).
    v_total : ndarray, shape (2*rank, nt) or (rank, nt)
        Matrix of right singular vectors (temporal coefficients).
    s_total : ndarray, shape (rank,)
        Vector of singular values.
    mode_list : list[int]
        Indices of the singular modes to visualise (0‑based).
    psd_multi : ndarray, shape (n_ranks, len(k_bins))
        Pre‑computed PSD curves for different truncation ranks.
    k_bins : ndarray
        Wavenumber bins corresponding to ''psd_multi''.
    labels : list[str] or None, optional
        Legend labels for each PSD curve.  If *None*, generic labels
        ``"PSD 1"``, ``"PSD 2"``… are used.
    nx, ny : int, optional
        Spatial grid dimensions used to reshape ``u_total``.  Default 256×256.
    nt : int, optional
        Number of time steps (x‑axis length for temporal plots).  Default 12 500.
    rank : int, optional
        Truncation rank that separates surface and velocity blocks in
        ``u_total`` / ``v_total`` when ``name != "surf"``.  Default 100.
    name : {"surf", "vel"}, optional
        Selects which half of the SVD matrices is plotted.  Determines the
        filename suffix.  Default ``"surf"``.

    Returns
    -------
    None
        The function produces a Matplotlib figure as a side‑effect and saves
        it to disk.

    Notes
    -----
    * Relies on the external ``ghibli`` palette for colours.
    * Assumes spatial modes are stored column‑wise: the first ``rank``
      columns correspond to surface (if ``name=="surf"``), the next
      ``rank`` columns to velocity.
    """
    
    import ghibli as gh

    # Set color palette and LaTeX rendering
    plt.rcParams['axes.prop_cycle'] = gh.ghibli_palettes['MononokeMedium']
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
    })

    # Decide which dataset to use
    if name== "surf":
        spatial_modes = u_total[:,:rank].reshape(nx,ny,rank)
        temporal_modes = v_total[:rank,:]
    else:
        spatial_modes = u_total[:, rank:2*rank].reshape(nx, ny, rank)
        temporal_modes = v_total[rank:2*rank, :]

    # Parameters for the figure
    time = np.linspace(0, nt, nt)
    nrows = 5
    ncolumns = 4
    #selected_indices = [0, 1, 2, 10, 25, 50, 100, 250]
    selected_indices = mode_list
    # Create the figure with flexible bottom row
    fig = plt.figure(figsize=(8, 10))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(nrows, ncolumns, height_ratios=[0.7, 0.7, 0.4, 0.4, 1.2], figure=fig)

    # Plot spatial modes in rows 0 and 1
    for i, idx in enumerate(selected_indices):
        row = i // ncolumns  # Row index
        col = i % ncolumns  # Column index
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(spatial_modes[:, :, idx], cmap='RdBu_r', interpolation='bilinear')
        ax.set_aspect('equal')  # Square aspect
        ax.axis('off')  # Remove axis

    # Plot temporal dynamics in rows 2 and 3
    ymin, ymax = np.min(temporal_modes[selected_indices]), np.max(temporal_modes[selected_indices])
    for i, idx in enumerate(selected_indices):
        row = 2 + (i // ncolumns)  # Temporal dynamics start at row 2
        col = i % ncolumns
        ax = fig.add_subplot(gs[row, col])
        ax.plot(time, temporal_modes[idx, :], color='black', linewidth=0.75)
        ax.set_xlim(0, nt)
        ax.set_ylim(ymin, ymax)
        if col == 0:  # Add ticks only for the first column
            ax.tick_params(axis='y', which='both', direction='out', length=3)
            ax.locator_params(axis='y', nbins=4)
        else:
            ax.yaxis.set_visible(False)
        if row == 3:  # Add x-axis ticks only for the last row
            ax.tick_params(axis='x', which='both', direction='out', length=3)
            ax.locator_params(axis='x', nbins=4)
        else:
            ax.xaxis.set_visible(False)

    # Bottom row: Log-log plots of singular values and power spectral density
    # ax1 = fig.add_subplot(gs[4, :1])
    # ax2 = fig.add_subplot(gs[4, 2:])
    ax1 = fig.add_axes([0.05, 0.1, 0.4, 0.24])  # [left, bottom, width, height]
    ax2 = fig.add_axes([0.55, 0.1, 0.4, 0.24])  # Adjust left to create a gap
    # Plot s_total

    ax1.semilogy(s_total[:len(s_total)-10]/s_total[0],label="Normalized Singular values") # second row, since we look at velocity here
   
    ax1.semilogy(np.cumsum(s_total[:len(s_total)-10] / sum(s_total)), label="Cumulative Sum", linestyle='--', color='#CD4F38')  # second row, since we look at velocity here
    ax1.set_xlabel('Rank')
    ax1.legend()
    #ax1.set_ylim(1e-3, 3)
    ax1.grid(True, which="both", linestyle="--", linewidth=0.3)
    # ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
    # ax1.set_ylabel("Normalized Singular Values")
    # Plot PSD results
    
    for i in range(psd_multi.shape[0]):
        #label = labels[i] if labels else f"PSD {i + 1}"
        label = f"r = {labels[i]}" if labels else f"PSD {i + 1}"
        plt.loglog(k_bins, psd_multi[i,:], label=label)
        #print(psd_multi[i,:])
    #ax2.set_xlim(10,10000)
    
    ax2.set_ylim(1e-8, 3)
    ax2.set_xlabel("$k L_{\infty}$", fontsize=13)
    ax2.set_ylabel("Normalized Power Spectral Density")
    ax2.legend()
    ax2.grid(True, which="both", linestyle="--", linewidth=0.3)

    # Add k^-5/3 slope
    k_slope = k_bins[(k_bins > 0) & (k_bins < np.max(k_bins))]
    slope = k_slope ** (-5 / 3)
    slope *= 6*psd_multi[0, 0] / slope[0]  # Use the second value to avoid zero-index issues
    ax2.loglog(k_slope, slope, 'k--', label='$k^{-5/3}$')


    # Adjust layout to minimize whitespace
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1, hspace=0.1, wspace=0.2)

    # Save the figure
    plt.savefig(f"svd_spectra_{name}.pdf", format="pdf", bbox_inches="tight", pad_inches=0.1)
    plt.show()



#done
def plot_svd_and_spectra_DNS(DNS_case,plane, rank_list, mode_list, labels):
    '''Loads and plots SVD modes and spectra for a specific plane of the DNS data
     
    Parameters
    ----------
    DNS_case : str
        DNS case identifier (e.g. "S1", "S2").
    plane : int
        Horizontal plane index (between 1-57 for S1, 1-76 for S2).
    rank_list : list[int]
        Sequence of SVD truncation ranks to compare.
    mode_list : list[int]
        Indices of singular modes to visualise (0‑based).
    labels : list[str]
        Legend labels for the different rank reconstructions.

    Returns
        None '''

    
    DNS_case = utilities.case_name_converter(DNS_case)

    #extract raw velocity field
    u_fluc = utilities.get_velocity_plane_DNS(DNS_case, plane)
    dimX, dimY, dimT = utilities.get_dims_DNS(DNS_case)

    #get SVD matrices from pre-calculated saved file:
    U, S, V = utilities.open_SVD(1000, ens=None, vel_fluc=False, variable='u', Teetank=False, teetank_case=None, forecast=False, DNS_new=True, DNS_plane=plane, 
                               DNS_surf=False, DNS_case=DNS_case, plane='H390')
    
    #get all singular values
    S_tot = utilities.get_singular_values_full(DNS_case, plane)
    
    #calculate PSD spectrum, to extract wavenumbers for binning
    spectra_ground_truth, k_bins = utilities.compute_psd(u_fluc, dx=1.0, dy=1.0)
    
    psd_multi_recon = np.zeros((len(rank_list), len(k_bins)))
    psd_multi_compr = np.zeros((len(rank_list), len(k_bins)))

    #calculate PSD spectrum for a range of rank values r given in the rank_list
    psd_multi_recon, k_vals = processdata.calculate_PSD_r_vals_DNS(DNS_case, u_fluc, rank_list, num_ens=1, plane=plane)

    #plot U and V modes for given modes in mode_list, as well as singular values and spectra for different ranks 
    plot_svd_and_spectra(U, np.transpose(V), S_tot, mode_list, psd_multi_recon, k_vals, labels = labels, nx=dimX, ny=dimY, nt=dimT, rank=1000, name="surf")



#done
def plot_svd_and_spectra_exp(exp_case,plane, rank_list, mode_list, labels, ensembles):
    '''Loads and plots SVD modes and spectra for a specific plane of the experimental data
     
    Parameters
    ----------
    exp_case : str
        Experiment identifier (e.g. "E1", "E2").
    plane : int
        Horizontal plane index (1 = H395, 2 = H390, ...).
    rank_list : list[int]
        Sequence of SVD truncation ranks to compare.
    mode_list : list[int]
        Indices of singular modes to visualise (0‑based).
    labels : list[str]
        Legend labels for the different rank reconstructions.
    ensembles : list[int]
        List of ensemble IDs to load; the first entry is used
        for raw‐field visualisation.

    Returns
        None '''


    exp_case = utilities.case_name_converter(exp_case)
    
    #extract raw velocity field
    u_fluc = utilities.get_velocity_plane_teetank(exp_case, plane)
    ensemble = ensembles[0]
    
    dimX, dimY, dimT = utilities.get_dims_teetank_vel()
    depths = ['H395', 'H390', 'H375', 'H350', 'H300']
    depth=depths[plane-1] #plane=1 is H395, plane=2 is H390 etc
    
    #open pre-calculated SVD matrices
    #NOTE: here we separate between matrices for velocity 'u' and surface elevation 'eta' due to difference in geometry/size
        #but time dynamics of same sampling and length, so the V matrices are stacked together
        #V_tot: matrix with horizontally stacked V matrices from eta (surface elevation field) and u (velocity field of chosen plane) respectively
    U_tot_u, S_tot_u, U_tot_surf, S_tot_surf, V_tot = utilities.open_SVD(900, ensemble, vel_fluc=False, variable='u', exp=True, experimental_case=exp_case, forecast=False, DNS_new=False, DNS_plane=None, DNS_surf=False, DNS_case=None, plane=depth)
    
    #exctract V matrix for velocity field, from the total stacked V matrix
    V_tot_u = V_tot[:, 900:]

    #calculate PSD spectrum, to extract wavenumbers for binning
    spectra_ground_truth, k_bins = utilities.compute_psd_1d(u_fluc[ensemble-1], dx=1e-3, dy=1e-3)

    psd_multi_recon = np.zeros((len(rank_list), len(k_bins)))
    psd_multi_compr = np.zeros((len(rank_list), len(k_bins)))

    #calculate PSD spectrum for a range of rank values r given in the rank_list
    psd_multi_recon, k_vals = processdata.calculate_PSD_r_vals_exp(u_fluc, rank_list, exp_case,  900, ensembles, depth)

    #plot U and V modes for given modes in mode_list, as well as singular values and spectra for different ranks 
    plot_svd_and_spectra(U_tot_u, np.transpose(V_tot_u), S_tot_u, mode_list, psd_multi_recon, k_vals, labels = labels, nx=dimX, ny=dimY, nt=dimT, rank=900, name="surf")




def plot_psd_compare(DNS_planes, exp_planes, SHRED_ensembles, experimental_ensembles, r_vals, num_sensors):

    #must open shred for each case
    

    planes = ['H395', 'H390', 'H375', 'H350', 'H300']


    #case E1 (P25)
        
    plane = planes[exp_planes[0]-1]
    X_surf, Y_surf, X_vel, Y_vel = utilities.get_mesh_exp('P25', plane)
    #open SHRED for this plane-surface-pairing, Tee-ensemble and SHRED ensemble
    V_tot_recons, V_tot_svd, test_indices = utilities.open_SHRED(experimental_ensembles[0], 'P25', r_vals[2], num_sensors, SHRED_ensembles[2], plane, DNS=False,  exp_plane=plane, full_planes=True, forecast=False)
    num_test_snaps = len(test_indices)
    #get SVDs correctly
    U_tot_u_red, S_tot_u_red, U_tot_surf_red, S_tot_surf_red, V_tot_red = utilities.open_and_reduce_SVD(experimental_ensembles[0], 'P25', 900, r_vals[2], forecast=False, DNS_new=False, DNS_plane=None,
                                                                                                                   DNS_surf=False, exp=True, exp_plane=plane)
    surf_fluc=None
    u_fluc_test, u_svd_test, u_recons_test, u_fluc_full = utilities.get_test_imgs_SHRED_exp(plane, surf_fluc, None, V_tot_recons, V_tot_svd, test_indices, X_surf, X_vel, experimental_ensembles[0], 'P25',900, r_vals[2], 
                                                                                                            SHRED_ensembles[2], num_sensors, U_tot_u_red, S_tot_u_red, V_tot_red = V_tot_red, open_svd=False, lags=52, forecast=False, 
                                                                                                            surface=False,no_input_u_fluc=True)
    gt_fft_p25, recon_fft_p25, k_vals_p25 = processdata.power_spectral_density_compare(u_fluc_test, u_recons_test, 3, DNS=False, DNS_case='RE2500')
    gt_fft, svd_fft_p25, k_vals_p25 = processdata.power_spectral_density_compare(u_fluc_test, u_svd_test, 3, DNS=False, DNS_case='RE2500')
    spectral_max_p25=np.amax(gt_fft_p25)
    gt_psd_p25 = gt_fft_p25/spectral_max_p25
    recon_psd_p25 = recon_fft_p25/spectral_max_p25
    svd_psd_p25 = svd_fft_p25/spectral_max_p25
    integral_length_scale=0.051
    k_vals_p25 = integral_length_scale*k_vals_p25
    k_cutoff_p25 = k_vals_p25[7]
    

    #case E2 (P50)

    plane = planes[exp_planes[1]-1]
    X_surf, Y_surf, X_vel, Y_vel = utilities.get_mesh_exp('P50', plane)
    #open SHRED for this plane-surface-pairing, Tee-ensemble and SHRED ensemble
    V_tot_recons, V_tot_svd, test_indices = utilities.open_SHRED(experimental_ensembles[1], 'P50', r_vals[3], num_sensors, SHRED_ensembles[3], plane, DNS=False,  plane=plane, full_planes=True, forecast=False)
    num_test_snaps = len(test_indices)
    #get SVDs correctly
    U_tot_u_red, S_tot_u_red, U_tot_surf_red, S_tot_surf_red, V_tot_red = utilities.open_and_reduce_SVD(experimental_ensembles[1], 'P50', 900, r_vals[3], forecast=False, DNS_new=False, DNS_plane=None,
                                                                                                                   DNS_surf=False, exp=True, Tee_plane=plane)
    surf_fluc=None
    u_fluc_test, u_svd_test, u_recons_test, u_fluc_full = utilities.get_test_imgs_SHRED_Teetank(plane, surf_fluc, None, V_tot_recons, V_tot_svd, test_indices, X_surf, X_vel, experimental_ensembles[1], 'P50',900, r_vals[3], 
                                                                                                            SHRED_ensembles[3], num_sensors, U_tot_u_red, S_tot_u_red, V_tot_red = V_tot_red, open_svd=False, lags=52, forecast=False, 
                                                                                                            surface=False,no_input_u_fluc=True)
    gt_fft_p50, recon_fft_p50, k_vals_p50 = processdata.power_spectral_density_compare(u_fluc_test, u_recons_test, 3, DNS=False, DNS_case='RE2500')
    gt_fft, svd_fft_p50, k_vals_p50 = processdata.power_spectral_density_compare(u_fluc_test, u_svd_test, 3, DNS=False, DNS_case='RE2500')
    spectral_max_p50=np.amax(gt_fft_p50)
    gt_psd_p50 = gt_fft_p50/spectral_max_p50
    recon_psd_p50 = recon_fft_p50/spectral_max_p50
    svd_psd_p50 = svd_fft_p50/spectral_max_p50
    integral_length_scale=0.068
    k_vals_p50 = integral_length_scale*k_vals_p50
    k_cutoff_p50 = k_vals_p50[7]

    #DNS case S1 (RE1000)

    stack_planes=[DNS_planes[0]] 
    plane = DNS_planes[0]
    integral_length_scale=utilities.get_integral_length_scale('RE1000')
    U_tot_red, S_tot_red, V_tot_red = processdata.stack_svd_arrays_DNS(stack_planes, r_vals[0],  DNS_case='RE1000')                                                  
    print("end stacking SVD")
    #then iterate SHRED ensembles
    V_tot_recons, V_tot_svd, test_indices = utilities.open_SHRED(None, 'RE1000', r_vals[0], num_sensors, SHRED_ensembles[0], stack_planes, DNS=True, 
                                                                     full_planes=True, forecast=False)
    u_fluc_test, u_svd_test, u_recons_test, u_fluc_full = utilities.get_test_imgs_SHRED_DNS('RE1000', plane, 1, None, V_tot_recons, test_indices, r_vals[0], 
                                                                                                    num_sensors,U_tot_red, S_tot_red, V_tot_red, open_svd=False, lags=52, 
                                                                                                    forecast=False, surface=False, no_input_u_fluc=True)
    gt_fft_RE1000, recon_fft_RE1000, k_vals_RE1000 = processdata.power_spectral_density_compare(u_fluc_test, u_recons_test, 3, DNS=True, DNS_case='RE1000')
    gt_fft, svd_fft_RE1000, k_vals_RE1000 = processdata.power_spectral_density_compare(u_fluc_test, u_svd_test, 3, DNS=True, DNS_case='RE1000')
    spectral_max_RE1000=np.amax(gt_fft_RE1000)
    gt_psd_RE1000 = gt_fft_RE1000/spectral_max_RE1000
    recon_psd_RE1000 = recon_fft_RE1000/spectral_max_RE1000
    svd_psd_RE1000 = svd_fft_RE1000/spectral_max_RE1000
    k_vals_RE1000 = integral_length_scale*k_vals_RE1000
    k_cutoff_RE1000 = k_vals_RE1000[13]
    
    #DNS case S2 (RE2500)
    
    stack_planes=[DNS_planes[1]] 
    plane = DNS_planes[1]
    U_tot_red, S_tot_red, V_tot_red = processdata.stack_svd_arrays_DNS(stack_planes, r_vals[1],  DNS_case='RE2500')                                                  
    print("end stacking SVD")
    #then iterate SHRED ensembles

    V_tot_recons, V_tot_svd, test_indices = utilities.open_SHRED(None, 'RE2500', r_vals[1], num_sensors, SHRED_ensembles[1], stack_planes, DNS=True, 
                                                                     full_planes=True, forecast=False)
 
    u_fluc_test, u_svd_test, u_recons_test, u_fluc_full = utilities.get_test_imgs_SHRED_DNS('RE2500', plane, 1, None, V_tot_recons, test_indices, r_vals[1], 
                                                                                                    num_sensors,U_tot_red, S_tot_red, V_tot_red, open_svd=False, lags=52, 
                                                                                                    forecast=False, surface=False, no_input_u_fluc=True)
    integral_length_scale=utilities.get_integral_length_scale('RE2500')
    gt_fft_RE2500, recon_fft_RE2500, k_vals_RE2500 = processdata.power_spectral_density_compare(u_fluc_test, u_recons_test, 3, DNS=True, DNS_case='RE2500')
    gt_fft, svd_fft_RE2500, k_vals_RE2500 = processdata.power_spectral_density_compare(u_fluc_test, u_svd_test, 3, DNS=True, DNS_case='RE2500')
    spectral_max_RE2500=np.amax(gt_fft_RE2500)
    gt_psd_RE2500 = gt_fft_RE2500/spectral_max_RE2500
    recon_psd_RE2500 = recon_fft_RE2500/spectral_max_RE2500
    svd_psd_RE2500 = svd_fft_RE2500/spectral_max_RE2500
    k_vals_RE2500 = integral_length_scale*k_vals_RE2500
    k_cutoff_RE2500 = k_vals_RE2500[11]


    #Plot figures
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.2, wspace=0.1)
    

    # Row 1: RMS for u and w
    ax1 = fig.add_subplot(gs[0, 0])
    #ax1.plot(test_snaps1, mse_snapshots_RE1000, label='MSE S1')
    #ax1.plot(test_snaps1, psd_snapshots_RE1000, label='PSD error S1')
    #ax1.plot(test_snaps1, ssim_snapshots_RE1000, label='SSIM S1')
    
    #ax2.plot(test_snaps1, psnr_snapshots_RE1000, color='r', label='PSNR S1')
    #ax2.tick_params(axis='y', labelcolor='r')
    ax1.loglog(k_vals_RE1000, gt_psd_RE1000, linestyle='-', color='k', label='Ground truth')
    ax1.loglog(k_vals_RE1000, svd_psd_RE1000, linestyle='dashdot', color='darkred', label='SVD')
    ax1.loglog(k_vals_RE1000, recon_psd_RE1000, linestyle='--', color='blue', label='Recons')
    ax1.plot([k_cutoff_RE1000, k_cutoff_RE1000], [1e-3,3], linestyle='-', color='darkviolet')
    ax1.set_ylabel('PSD', fontsize=15)
    ax1.grid()
    ax1.legend(fontsize=13)
    ax1.set_ylim(1e-3, 3)
    ax1.set_xlim(k_vals_RE1000[0], 11)
    ax1.set_xlabel("$k L_{\infty}$", fontsize=15)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.loglog(k_vals_RE2500, gt_psd_RE2500, linestyle='-', color='k', label='Ground truth')
    ax2.loglog(k_vals_RE2500, svd_psd_RE2500, linestyle='dashdot', color='darkred', label='SVD')
    ax2.loglog(k_vals_RE2500, recon_psd_RE2500, linestyle='--', color='blue', label='Recons')
    ax2.plot([k_cutoff_RE2500, k_cutoff_RE2500], [1e-3,3], linestyle='-', color='darkviolet')
    #ax2.set_ylabel('PSD', fontsize=15)
    ax2.grid()
    ax2.legend(fontsize=13)
    ax2.set_ylim(1e-3, 3)
    ax2.set_xlim(k_vals_RE2500[0], 11)
    ax2.set_xlabel("$k L_{\infty}$", fontsize=15)



    ax3 = fig.add_subplot(gs[1,0])
    ax3.loglog(k_vals_p25, gt_psd_p25, linestyle='-', color='k', label='Ground truth')
    ax3.loglog(k_vals_p25, svd_psd_p25, linestyle='dashdot', color='darkred', label='SVD')
    ax3.loglog(k_vals_p25, recon_psd_p25, linestyle='--', color='blue', label='Recons')
    ax3.plot([k_cutoff_p25, k_cutoff_p25], [1e-3,3], linestyle='-', color='darkviolet')
    ax3.set_ylabel('PSD', fontsize=15)
    ax3.grid()
    ax3.legend(fontsize=13)
    ax3.set_ylim(1e-3, 3)
    ax3.set_xlim(k_vals_p25[0], 11)
    ax3.set_xlabel("$k L_{\infty}$", fontsize=15)

    ax4= fig.add_subplot(gs[1,1])
    ax4.loglog(k_vals_p50, gt_psd_p50, linestyle='-', color='k', label='Ground truth')
    ax4.loglog(k_vals_p50, svd_psd_p50, linestyle='dashdot', color='darkred', label='SVD')
    ax4.loglog(k_vals_p50, recon_psd_p50, linestyle='--', color='blue', label='Recons')
    ax4.plot([k_cutoff_p50, k_cutoff_p50], [1e-3,3], linestyle='-', color='darkviolet')
    #ax4.set_ylabel('PSD', fontsize=15)
    ax4.grid()
    ax4.legend(fontsize=13)
    ax4.set_ylim(1e-3, 3)
    ax4.set_xlim(k_vals_p50[0], 11)
    ax4.set_xlabel("$k L_{\infty}$", fontsize=15)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(axis='x', which='major', labelsize=13)
        ax.tick_params(axis='y', which='major', labelsize=13)

    ax2.tick_params(left=False, labelleft=False)
    ax4.tick_params(left=False, labelleft=False)

    fig.tight_layout()
    plt.show()
    adr_loc_3 = "C:\\Users\krissmoe\OneDrive - NTNU\PhD\PhD code\PhD-1\Flow Reconstruction and SHRED\Results\\psd_compare" 
    filename = adr_loc_3
    filename2 = filename + ".png"
    fig.savefig(filename2)
    fig.savefig(filename+".pdf", format='pdf', bbox_inches='tight', pad_inches=0.1) 



def plot_recon_compare(surf_height, flow_surf, flow_depth, rank, nt):
    """
    Compare snapshots from the reconstructed data with original data and compressed data
    """
    #import matplotlib.pyplot as plt
    import cmocean

    row1 = [surf_height[0], surf_height[1], surf_height[2]]  # First row
    row2 = [flow_surf[0], flow_surf[1], flow_surf[2]]  # Second row
    row3 = [flow_depth[0], flow_depth[1], flow_depth[2]]  # Third row
    #L_int_ind = 5  # Example index for flow slices
    # Organize data into rows for easy plotting
    #row1 = [surf_full, surf_compr, surf_recon]
    #row1 = [surf_height[:, :, 0], flow_compr[:, :, 0], flow_recon[:, :, 0]]
    #row2 = [flow_full[:, :, 0], flow_compr[:, :, 0], flow_recon[:, :, 0]]
    #row2 = [flow_full[:, :, 2], flow_compr[:, :, 2], flow_recon[:, :, 2]]
    #row3 = [flow_full[:, :, L_int_ind], flow_compr[:, :, L_int_ind], flow_recon[:, :, L_int_ind]]
    all_rows = [row1, row2, row3]

    # Plot each snapshot in the grid
    spacing = 0.1
    fig, axs = plt.subplots(3, 3, figsize=(15, 15),  gridspec_kw={'wspace': spacing, 'hspace': spacing})
    #fig.subplots_adjust(wspace=spacing, hspace=spacing)
    cmaps = [cmocean.cm.ice,'RdBu_r','RdBu_r'] # Alternative colors for velocity: cm.thermal, cm. balance
    for i, row in enumerate(all_rows):
        for j, snapshot in enumerate(row):
            ax = axs[i, j]
            ax.imshow(snapshot, cmap=cmaps[i], interpolation='bilinear', aspect='auto')
            ax.axis('off')

    # Adjust layout and display the plot
   # plt.tight_layout()
    plt.show()

    plt.savefig("compare_recon_rank"+str(rank)+"_nt"+str(nt)+".eps", format='eps', bbox_inches='tight', pad_inches=0.5)




def plot_psd_comparison(psd_recon, psd_compr, psd_ground_truth, k_bins, labels_recon=None, labels_compr=None, split_rank=0, rank_list=None):
    """
    Plots PSD comparisons for reconstructed and compressed data along with ground truth.

    Parameters:
    ----------
    psd_recon : numpy.ndarray
        Array of PSDs from reconstructed data (shape: [n_psds, n_frequencies]).
    psd_compr : numpy.ndarray
        Array of PSDs from compressed data (shape: [n_psds, n_frequencies]).
    psd_ground_truth : numpy.ndarray
        Array of the ground truth PSD (shape: [n_frequencies]).
    k_bins : numpy.ndarray
        Array of wavenumber bins (shape: [n_frequencies]).
    labels_recon : list of str
        Labels for each PSD in psd_recon (optional).
    labels_compr : list of str
        Labels for each PSD in psd_compr (optional).
    split_rank : cutoff rank for curves in left frame
    """
    # Color map and LaTeX properties
    import ghibli as gh
    plt.rcParams['axes.prop_cycle'] = gh.ghibli_palettes['MononokeMedium']
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
    })

    # Sanity check for input consistency
    assert psd_recon.shape == psd_compr.shape, "psd_recon and psd_compr must have the same shape"
    assert psd_recon.shape[1] == len(k_bins), "k_bins length must match the number of frequencies"
    n_psds = psd_recon.shape[0]

    if split_rank:
        split_idx = len([i for i in rank_list if i < split_rank])
    else:
        split_idx = n_psds // 2  # Index to split the data for two panels

    # Create a figure with two subplots (panels)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # Define function to use for insets in the plots
    def add_inset(ax, psd_recon, psd_compr, psd_ground_truth, k_bins, start_idx, end_idx, color_list):
        """
        Adds a zoomed-in inset to the given subplot.
        """
        inset = inset_axes(ax, width="35%", height="35%", loc='lower left',
                           bbox_to_anchor=(0.1, 0.1, 1, 1), bbox_transform=ax.transAxes)
        next(ax._get_lines.prop_cycler)['color']
        for i in range(start_idx, end_idx):
            print(start_idx, end_idx)
            print(i-start_idx, len(color_list))
            color = color_list[i-start_idx]#next(ax._get_lines.prop_cycler)['color']
            inset.plot(k_bins[:2], psd_recon[i, :2], linestyle='-', linewidth=1.5, color=color)
            inset.plot(k_bins[:2], psd_compr[i, :2], linestyle='--', linewidth=1.5, color=color)
        inset.plot(k_bins[:2], psd_ground_truth[:2], linestyle='-.', linewidth=2, color='black')
        inset.set_xlim(k_bins[0], k_bins[1])
        inset.set_ylim(min(psd_recon[:, :2].min(), psd_compr[:, :2].min(), psd_ground_truth[:2].min()) * 0.8,
                       max(psd_recon[:, :2].max(), psd_compr[:, :2].max(), psd_ground_truth[:2].max()) * 1.2)
        inset.set_xscale('log')
        inset.set_yscale('log')
        inset.tick_params(axis='both', which='both', labelsize=10)
        inset.grid(True, which="both", linestyle="--", linewidth=0.3)

    # Panel 1: First half of PSDs
    next(axs[0]._get_lines.prop_cycler)['color'] # Skip first color
    color_list = []
    for i in range(split_idx):
        color = next(axs[0]._get_lines.prop_cycler)['color']
        color_list.append(color)
        label_recon = labels_recon[i] if labels_recon else f"Reconstructed {i+1}"
        label_compr = labels_compr[i] if labels_compr else f"Compressed {i+1}"
        axs[0].loglog(k_bins, psd_recon[i, :], label=label_recon, linestyle='-', linewidth=1.5, color=color)
        axs[0].loglog(k_bins, psd_compr[i, :], label=label_compr, linestyle='--', linewidth=1.5, color=color)

    axs[0].loglog(k_bins, psd_ground_truth, label="Ground truth",linestyle='-.', linewidth=2, color='black')
    axs[0].set_ylim(1e-8, 2)
    axs[0].set_xlabel("Dimensionless Wavenumber $kL$")
    axs[0].set_ylabel("Normalized Power Spectral Density")
    axs[0].legend()
    axs[0].grid(True, which="both", linestyle="--", linewidth=0.3)

    # Add inset to the first panel
    add_inset(axs[0], psd_recon, psd_compr, psd_ground_truth, k_bins, 0, split_idx, color_list)

    # Panel 2: Second half of PSDs
    next(axs[1]._get_lines.prop_cycler)['color'] # Skip first color
    color_list = []
    for i in range(split_idx, n_psds):
        color = next(axs[1]._get_lines.prop_cycler)['color']
        color_list.append(color)
        label_recon = labels_recon[i] if labels_recon else f"Reconstructed {i+1}"
        label_compr = labels_compr[i] if labels_compr else f"Compressed {i+1}"
        axs[1].loglog(k_bins, psd_recon[i, :], label=label_recon,linestyle='-', linewidth=1.5, color=color)
        axs[1].loglog(k_bins, psd_compr[i, :], label=label_compr,linestyle='--', linewidth=1.5, color=color)

    axs[1].loglog(k_bins, psd_ground_truth, label="Ground truth",linestyle='-.', linewidth=2, color='black')
    axs[1].set_ylim(1e-8, 2)
    axs[1].set_xlabel("Dimensionless Wavenumber $kL$")
    axs[1].set_ylabel("Normalized Spectral Density")
    axs[1].legend()
    axs[1].grid(True, which="both", linestyle="--", linewidth=0.3)

    # Add inset to the second panel
    add_inset(axs[1], psd_recon, psd_compr, psd_ground_truth, k_bins, split_idx, n_psds, color_list)

            
    # Save and display the plot
    plt.savefig("psd_comparison.eps", format='eps', bbox_inches='tight', pad_inches=0.1)
    plt.show()




def plot_depth_dependent_error_metrics(DNS_cases, exp_cases, ranks, colors, vel_planes_S1, vel_planes_S2, vel_planes_exp, num_sensors, SHRED_ensembles_S1, SHRED_ensembles_S2, SHRED_ensembles_E1, SHRED_ensembles_E2, exp_ensembles_E1, exp_ensembles_E2, forecast=False,full_planes=True, full_planes_exp=False, z_norm=None):
    """
    Build the “four-case” figure that compares depth-dependent SHRED
    reconstruction metrics for

    * DNS case S1 (`DNS_cases[0]`)
    * DNS case S2 (`DNS_cases[1]`)
    * Experimental case E1 (`exp_cases[0]`)
    * Experimental case E2 (`exp_cases[1]`)


    Parameters
    ----------
    DNS_cases : tuple[str, str]
        Identifiers for the two DNS datasets, e.g. ``(S1, S2)``.
    exp_cases : tuple[str, str]
        Identifiers for the two experimental datasets, e.g. ``(E1, E2)``.
    ranks : tuple[int, int, int, int]
        SVD truncation ranks used for S1, S2, E1, E2 (in that order).
    colors : tuple[str, str, str, str]
        Line colours for the four cases when plotting.
    vel_planes1, vel_planes2 : list[int]
        Velocity-plane indices used in S1 and S2 metric calculations.
    vel_planes_exp : list[int]
        Velocity-plane indices common to both experimental cases.
    num_sensors : int
        Number of surface sensors in all SHRED runs.
    SHRED_ensembl_S1, SHRED_ensembles_S2 : list[int]
        SHRED ensemble seeds for S1 and S2.
    SHRED_ensembles_E1, SHRED_ensembles_E2 : list[int]
        SHRED ensemble seeds for E1 and E2.
    exp_ensembles_E1, exp_ensembles_E2 : list[int]
        Teetank ensemble indices providing experimental SVDs.
    forecast : bool, default False
        If *True*, load forecast-mode SHRED outputs.
    full_planes : bool, default True
        If *True*, DNS SHRED models were trained on all planes.
    full_planes_exp : bool, default False
        If *True*, experimental SHRED models used all planes; otherwise
        just ``vel_planes_exp``.
    z_norm : {"int", "depth", None}, optional
        Normalisation for the vertical axis: integral length scale,
        physical depth, or *None* (raw index).

    Returns
    -------
    None
        Generates a Matplotlib figure (via
        :pyfunc:`plot_error_metrics_four_cases`) but does not return data.


    """

    #get necessary profiles of RMS and depth values
    DNS_case1 = utilities.case_name_converter(DNS_cases[0])
    DNS_case2 = utilities.case_name_converter(DNS_cases[1])
    exp_case1 = utilities.case_name_converter(exp_cases[0])
    exp_case2 = utilities.case_name_converter(exp_cases[1])

    
    #RMS for DNS cases S1 and S2
    #requires these have been calculated by 'calc_RMS_profile_true()' and stored as mat files
    RMS_z1 = processdata.get_RMS_profile_true(DNS_case1, vel_planes_S1) #returns RMS_profile with only chosen planes
    RMS_z2 = processdata.get_RMS_profile_true(DNS_case2, vel_planes_S2)

    #RMS for experimental cases E1 and E2 
    RMS_exp_z1 = processdata.get_RMS_profile_true_exp(vel_planes_exp, exp_case1, None, experimental_ens_avg=True)
    RMS_exp_z2 = processdata.get_RMS_profile_true_exp(vel_planes_exp, exp_case2, None, experimental_ens_avg=True)
    RMS_exp_z1 = RMS_exp_z1[1:]
    RMS_exp_z2 = RMS_exp_z2[1:]

    #get vertical axis
    z1 = utilities.get_zz_DNS(DNS_case1)
    z2 = utilities.get_zz_DNS(DNS_case2)
    z_exp = utilities.get_zz_exp()

    #normalize vertical axis with a length scale set by 'z_norm' (for Fig. 9 in paper: 'int' for turbulent integral length scale)
    z1 = utilities.get_normalized_z(z1, z_norm, DNS_case1)
    z2 = utilities.get_normalized_z(z2, z_norm, DNS_case2)
    z1_exp = utilities.get_normalized_z_exp(z_exp, z_norm, exp_case1)
    z2_exp = utilities.get_normalized_z_exp(z_exp, z_norm, exp_case2)
    z1_exp = z1_exp[1:]
    z2_exp = z2_exp[1:]


    #exctract ensemble-averaged error metrics
    RMS_recons_avg_1, RMS_true_avg, mse_avg_1, ssim_avg_1, psnr_avg_1, psd_avg_1, std_RMS_recons_1, std_mse_z, std_ssim, std_psnr, std_psd= processdata.get_ensemble_avg_error_metrics(DNS_case1,
        ranks[0], vel_planes_S1, num_sensors, SHRED_ensembles_S1, forecast=forecast, full_planes=full_planes)
    
    RMS_recons_avg_2, RMS_true_avg, mse_avg_2, ssim_avg_2, psnr_avg_2, psd_avg_2, std_RMS_recons_2, std_mse_z, std_ssim, std_psnr, std_psd= processdata.get_ensemble_avg_error_metrics(DNS_case2,
        ranks[1], vel_planes_S2, num_sensors, SHRED_ensembles_S2, forecast=forecast, full_planes=full_planes)
    

    #extract ensemble-avg error metrics from experiments
    RMS_recons_avg_exp1, RMS_true_avg_exp1, mse_avg_exp1, ssim_avg_exp1, psnr_avg_exp1, psd_avg_exp1, std_RMS_recons_exp1, std_mse_z, std_ssim, std_psnr, std_psd= processdata.get_ensemble_avg_error_metrics_exp(exp_case1, 
        ranks[2], vel_planes_exp, num_sensors, SHRED_ensembles_E1, exp_ensembles_E1, lags=52, forecast=False, full_planes=full_planes_exp)

    RMS_recons_avg_exp2, RMS_true_avg_exp2, mse_avg_exp2, ssim_avg_exp2, psnr_avg_exp2, psd_avg_exp2, std_RMS_recons_exp2, std_mse_z, std_ssim, std_psnr, std_psd= processdata.get_ensemble_avg_error_metrics_exp(exp_case2, 
        ranks[3], vel_planes_exp, num_sensors, SHRED_ensembles_E2, exp_ensembles_E2, lags=52, forecast=False, full_planes=full_planes_exp)

    std_RMS_recons_exp1=(1/np.sqrt(len(exp_ensembles_E1)))*std_RMS_recons_exp1 # np.zeros(5)
    std_RMS_recons_exp2=(1/np.sqrt(len(exp_ensembles_E2)))*std_RMS_recons_exp2

    #the plotter for all four cases S1, S2, E1, E2, all together
    plot_error_metrics_four_cases(DNS_case1, DNS_case2, exp_case1, exp_case2, None, colors,
    z1, z2, z1_exp, z2_exp, RMS_z1, RMS_z2, RMS_exp_z1, RMS_exp_z2, RMS_recons_avg_1, RMS_recons_avg_2, RMS_recons_avg_exp1, RMS_recons_avg_exp2, 
    std_RMS_recons_1, std_RMS_recons_2, std_RMS_recons_exp1, std_RMS_recons_exp2, mse_avg_1, mse_avg_2, mse_avg_exp1, mse_avg_exp2, psd_avg_1, psd_avg_2, psd_avg_exp1, psd_avg_exp2, 
    ssim_avg_1, ssim_avg_2, ssim_avg_exp1, ssim_avg_exp2, psnr_avg_1, psnr_avg_2, psnr_avg_exp1, psnr_avg_exp2, fig=None, gs=None, z_norm=z_norm)


        
def plot_error_metrics_per_case(DNS_case, k_index, color_curve,
    depth, rms_u_gt, rms_u_recon, std_rms_u, rms_w_gt, rms_w_recon,
    rms_vort_x_gt, rms_vort_x_recon, rms_vort_z_gt, rms_vort_z_recon,
    nmse_data, std_nmse, psd_data, std_psd, ssim_data, std_ssim,  psnr_data, std_psnr, fig=None, gs=None
):
    #TODO: Make plots here to error plots, remove mid panels
    #also make another function that calculates the ensembled averages and std_values
    #also a function to extract the right z values in z axis given a plane
    """
    Plots depth-dependent error metrics in an 8-panel layout.

    Parameters:
        depth: array-like, depth values (z-axis, positive downward).
        rms_u_gt, rms_u_recon: RMS of u-velocity for ground truth and reconstruction.
        rms_w_gt, rms_w_recon: RMS of w-velocity for ground truth and reconstruction (can be None).
        rms_vort_x_gt, rms_vort_x_recon: RMS vorticity (x-component) for ground truth and reconstruction (can be None).
        rms_vort_z_gt, rms_vort_z_recon: RMS vorticity (z-component) for ground truth and reconstruction (can be None).
        nmse_data: Normalized Mean Squared Error.
        psd_data: Power Spectral Density Error.
        ssim_data: Structural Similarity Index Measure.
        psnr_data: Peak Signal-to-Noise Ratio.
        std_xxx : standard deviation for the ensembled signals
    """
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
    })
    # Helper function for single-curve panels
    def plot_single_curve(ax, data, depth, err, label, xlabel, color_curve):

        lower_bound = data - err
        upper_bound = data + err

        if len(data)>30:
            marker1=None
            ax.plot(data, depth, linestyle='-', marker=marker1, color=color_curve, label=label)
            #ax.fill_betweenx(depth, lower_bound, upper_bound,color=color_curve,alpha=0.2)
            
        else:
            marker1='o'
            ax.errorbar(data, depth, xerr = err, linestyle='-', marker=marker1, label=label, capsize=3, errorevery=3)
        
        
        
        #ax.plot(data, depth, linestyle='-', marker='o', label=label)
        #ax.invert_yaxis()
        ax.set_xlabel(xlabel)

    # Helper function for two-curve panels
    def plot_two_curves(ax, data1, data2, depth, err2, labels, xlabel, color_curve):

        lower_bound = data2 - err2
        upper_bound = data2 + err2
        if len(data2)>30:
            marker1=None
            marker2=None
            ax.plot(data2, depth, linestyle='-', marker=marker2, color=color_curve, label=labels[1])
            ax.fill_betweenx(depth, lower_bound, upper_bound,color=color_curve,alpha=0.2)
        else:
            marker2='s'
            marker1 ='o'
            ax.errorbar(data2, depth, xerr = err2, linestyle='-', marker=marker1, label=labels[1], capsize=3, errorevery=1)



        #ax.errorbar(data2, depth, xerr = err2, linestyle='-', marker=marker1, label=labels[1], capsize=3, errorevery=1)
    
        ax.plot(data1, depth, linestyle='--', marker=marker1, color=color_curve, label=labels[0])
        
       
        #ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.legend()

    # Create a figure and use GridSpec for layout
    if fig==None:
        fig = plt.figure(figsize=(14, 15))
        gs = GridSpec(3, 2, figure=fig, hspace=0.2, wspace=0.1)
    

    # Row 1: RMS for u and w
    ax1 = fig.add_subplot(gs[0, 0])
    plot_two_curves(ax1, rms_u_gt, rms_u_recon, depth, err2=std_rms_u, labels=['Ground Truth' +DNS_case, 'Recons' + DNS_case], xlabel='u RMS ', color_curve=color_curve)

    ax2 = fig.add_subplot(gs[0, 1])
    if rms_w_gt is not None and rms_w_recon is not None:
        print("REMEMBER TO FIX ERROR INPUT")
        plot_two_curves(ax2, rms_w_gt, rms_w_recon, depth, labels=['Ground Truth' + DNS_case, 'Recons' + DNS_case], xlabel='None', color_curve=color_curve)
    else:
        ax2.text(0.5, 0.5, 'To be filled/removed', ha='center', va='center', transform=ax2.transAxes)
        ax2.invert_yaxis()
        ax2.set_xlabel('RMS')

    # Row 2: RMS Vorticity for x and z components
    #ax3 = fig.add_subplot(gs[1, 0])
    #if rms_vort_x_gt is not None and rms_vort_x_recon is not None:
    #    plot_two_curves(ax3, rms_vort_x_gt, rms_vort_x_recon, depth, labels=['Ground Truth', 'Reconstruction'], xlabel='RMS Vorticity (x)')
    #else:
    #    ax3.text(0.5, 0.5, 'Vorticity not computed', ha='center', va='center', transform=ax3.transAxes)
    #    ax3.invert_yaxis()
    #    ax3.set_xlabel('RMS Vorticity (x)')

    #ax4 = fig.add_subplot(gs[1, 1])
    #if rms_vort_z_gt is not None and rms_vort_z_recon is not None:
    #    plot_two_curves(ax4, rms_vort_z_gt, rms_vort_z_recon, depth, labels=['Ground Truth', 'Reconstruction'], xlabel='RMS Vorticity (z)')
    #else:
    #    ax4.text(0.5, 0.5, 'Vorticity not computed', ha='center', va='center', transform=ax4.transAxes)
    #    ax4.invert_yaxis()
    #    ax4.set_xlabel('RMS Vorticity (z)')

    # Row 3: NMSE and PSD Error
    ax5 = fig.add_subplot(gs[1, 0])
    plot_single_curve(ax5, nmse_data, depth, err=std_nmse, label=DNS_case, xlabel='Normalized Mean Squared Error', color_curve=color_curve)

    ax6 = fig.add_subplot(gs[1, 1])
    plot_single_curve(ax6, psd_data, depth, err=std_psd, label=DNS_case, xlabel='Power Spectral Density Error', color_curve=color_curve)

    # Row 4: SSIM and PSNR
    ax7 = fig.add_subplot(gs[2, 0])
    plot_single_curve(ax7, ssim_data, depth, err=std_ssim, label=DNS_case, xlabel='SSIM', color_curve=color_curve)

    ax8 = fig.add_subplot(gs[2, 1])
    plot_single_curve(ax8, psnr_data, depth, err=std_psnr, label=DNS_case, xlabel='PSNR', color_curve=color_curve)

    # Adjust layout

    if k_index==1:
        fig.tight_layout()
        plt.show()
        adr_loc_3 = "C:\\Users\krissmoe\OneDrive - NTNU\PhD\PhD code\PhD-1\Flow Reconstruction and SHRED\Results\\DNS_" + DNS_case
        filename = adr_loc_3 + "error_metrics"
        filename2 = filename + ".png"
        fig.savefig(filename2)
        fig.savefig(filename+".pdf", format='pdf', bbox_inches='tight', pad_inches=0.1)
    return fig, gs


def plot_error_metrics_four_cases(DNS_case1, DNS_case2, teetank_case1, teetank_case2, k_index, colors,
    depth1, depth2, depth3, depth4, rms_u_gt_1, rms_u_gt_2, rms_u_gt_3, rms_u_gt_4, rms_u_recon_1, rms_u_recon_2, rms_u_recon_3, rms_u_recon_4, 
    std_rms_u_1, std_rms_u_2, std_rms_u_3, std_rms_u_4,nmse_data_1, nmse_data_2, nmse_data_3, nmse_data_4, psd_data_1, psd_data_2, psd_data_3, psd_data_4, 
    ssim_data_1, ssim_data_2, ssim_data_3, ssim_data_4, psnr_data_1, psnr_data_2, psnr_data_3, psnr_data_4, fig=None, gs=None, z_norm=None
):

    """
    Plots depth-dependent error metrics in an 8-panel layout.

    Parameters:
        depth: array-like, depth values (z-axis, positive downward).
        rms_u_gt, rms_u_recon: RMS of u-velocity for ground truth and reconstruction.
        rms_w_gt, rms_w_recon: RMS of w-velocity for ground truth and reconstruction (can be None).
        rms_vort_x_gt, rms_vort_x_recon: RMS vorticity (x-component) for ground truth and reconstruction (can be None).
        rms_vort_z_gt, rms_vort_z_recon: RMS vorticity (z-component) for ground truth and reconstruction (can be None).
        nmse_data: Normalized Mean Squared Error.
        psd_data: Power Spectral Density Error.
        ssim_data: Structural Similarity Index Measure.
        psnr_data: Peak Signal-to-Noise Ratio.
        std_xxx : standard deviation for the ensembled signals
    """
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
    })

    #make z_label:
    if z_norm=='taylor':
        z_label='$z/L_{\lambda}$'
    elif z_norm == 'int':
        z_label='$z/L_{\infty}$'
    elif z_norm=='mixed':
        z_label='$z/(L_{\lambda} L_{\infty})^{1/2}$'  
    else:
        z_label = None
      
    # Helper function for single-curve panels
    def plot_single_curve(ax, data1, data2, data3, data4, depth1, depth2, depth3, depth4, err, label1, label2, label3, label4, xlabel, colors, xlim=False, z_label=None, z_tics=True):


        if len(data1)>30:
            marker1=None
            ax.plot(data1, depth1, linestyle='-', marker=marker1, color=colors[0], label=label1)
            ax.plot(data2, depth2, linestyle='-', marker=marker1, color=colors[1], label=label2)
            
            #activate when using P25
            ax.plot(data3, depth3, marker='o', color=colors[2], label=label3)
            ax.plot(data4, depth4, marker='o', color=colors[3], label=label4)

            #ax.fill_betweenx(depth, lower_bound, upper_bound,color=color_curve,alpha=0.2)
            if z_label != None:
                ax.set_ylabel(z_label, fontsize=22)
        else:
            marker1='o'
            ax.plot(data1, depth1, linestyle='-', marker=marker1, color=colors[0], label=label1)
            ax.plot(data2, depth2, marker=marker1, color=colors[1], label=label2)
            #ax.tick_params(axis='both', which='major', labelsize=14, length=10)
            #ax.errorbar(data1, depth1, xerr = err, linestyle='-', marker=marker1, label=label, capsize=3, errorevery=3)
        
        if z_tics:
            ax.tick_params(axis='both', which='major', labelsize=16, length=10)
        else:
            #ax.tick_params(axis='both', which='major', labelsize=14, length=10)
            ax.tick_params(axis='y', which='both', labelsize=0, length=0)  # Hides ticks on y-axis
            ax.tick_params(axis='x', which='major', labelsize=16, length=10)
        
        #ax.plot(data, depth, linestyle='-', marker='o', label=label)
        #ax.invert_yaxis()
        ax.set_xlabel(xlabel, fontsize=15)
        if xlim:
            ax.set_xlim(0.0)
        ax.legend(fontsize=16)
        ax.grid()

    # Helper function for two-curve panels
    def plot_two_curves(ax, data1_truth, data1_recons, data2_truth, data2_recons, depth1, depth2,
                         err1_recons, err2_recons,labels1, labels2, xlabel, colors, DNS=True, z_label=None, z_tics=True):

        
        lower_bound1 = data1_recons - err1_recons
        upper_bound1 = data1_recons + err1_recons
        lower_bound2 = data2_recons - err2_recons
        upper_bound2 = data2_recons + err2_recons
        
        if DNS:
            marker1=None
            marker2=None
            ax.plot(data1_recons, depth1, linestyle='dashdot', marker=marker2, color=colors[0], label=labels1[1])
            ax.fill_betweenx(depth1, lower_bound1, upper_bound1,color=colors[0],alpha=0.2)
            ax.plot(data1_truth, depth1, linestyle='-', marker=marker1, color=colors[0], label=labels1[0])
            ax.plot(data2_recons, depth2, linestyle='dashdot', marker=marker2, color=colors[1], label=labels2[1])
            ax.fill_betweenx(depth2, lower_bound2, upper_bound2,color=colors[1],alpha=0.2)
            ax.plot(data2_truth, depth2, linestyle='-', marker=marker1, color=colors[1], label=labels2[0])
            if z_label != None:
                ax.set_ylabel(z_label, fontsize=21)
            #ax.tick_params(axis='both', which='major', labelsize=14, length=10)
        
        else:
            #marker2='s'
            #marker1 ='o'
            #ax.errorbar(data2, depth, xerr = err2, linestyle='-', marker=marker1, label=labels[1], capsize=3, errorevery=1)
            #activate when P25 is ready
            ax.plot(data1_truth, depth1, marker='D', color=colors[2], label=labels1[0])
            ax.plot(data2_truth, depth2, marker='D', color=colors[3], label=labels2[0])
            ax.errorbar(data1_recons, depth1, xerr = err1_recons, linestyle='dashdot', fmt='o', color=colors[2],label=labels1[1], capsize=2, errorevery=1)
            ax.errorbar(data2_recons, depth2, xerr=err2_recons, linestyle='dashdot', fmt='o', color=colors[3],label=labels2[1], capsize=2, errorevery=1)
            #ax.tick_params(axis='both', which='major', labelsize=14, length=10)
            #ax.plot(data1_truth, depth1, marker='D', color=colors[2], label=labels1[0])
            #ax.plot(data2_truth, depth2, marker='D', color=colors[3], label=labels2[0])
            if z_label != None:
                ax.set_ylabel(z_label, fontsize=21)
        if z_tics:
            ax.tick_params(axis='both', which='major', labelsize=16, length=10)
        else:
            #ax.tick_params(axis='both', which='major', labelsize=14, length=10)
            ax.tick_params(axis='y', which='both', labelsize=0, length=0)  # Hides ticks on y-axis
            ax.tick_params(axis='x', which='major', labelsize=16, length=10)
        
        ax.set_xlim(0.0)
        ax.set_ylim(-2.8,0.05)
        ax.grid()

        #ax.errorbar(data2, depth, xerr = err2, linestyle='-', marker=marker1, label=labels[1], capsize=3, errorevery=1)
    
        #ax.plot(data1_truth, depth1, linestyle='--', marker=marker1, color=colors[0], label=labels1[0])
        #ax.plot(data2_truth, depth2, linestyle='--', marker=marker1, color=colors[1], label=labels2[0])
        
       
        #ax.invert_yaxis()
        ax.set_xlabel(xlabel, fontsize=18)
        ax.legend(fontsize=16)

    # Create a figure and use GridSpec for layout
    if fig==None:
        fig = plt.figure(figsize=(14, 15))
        gs = GridSpec(3, 2, figure=fig, hspace=0.2, wspace=0.1)
    

    # Row 1: RMS for u and w
    ax1 = fig.add_subplot(gs[0, 0])
    labels1=['Ground Truth S1', 'Recons S2']
    labels2=['Ground Truth S2', 'Recons S2']
    plot_two_curves(ax1, rms_u_gt_1, rms_u_recon_1, rms_u_gt_2, rms_u_recon_2, depth1, depth2, err1_recons=std_rms_u_1, err2_recons=std_rms_u_2,
                     labels1=labels1, labels2=labels2, xlabel='(a) RMS u cases S1 \& S2', colors=colors, DNS=True, z_label=z_label)

    ax2 = fig.add_subplot(gs[0, 1])
    #if rms_w_gt is not None and rms_w_recon is not None:
    #    print("REMEMBER TO FIX ERROR INPUT")
    #    plot_two_curves(ax2, rms_w_gt, rms_w_recon, depth, labels=['Ground Truth' + DNS_case, 'Recons' + DNS_case], xlabel='None', color_curve=color_curve, z_label=z_label)
    labels3=['Ground Truth E1', 'Recons E1']
    labels4=['Ground Truth E2', 'Recons E2']
    
    plot_two_curves(ax2, rms_u_gt_3, rms_u_recon_3, rms_u_gt_4, rms_u_recon_4, depth3, depth4, err1_recons=std_rms_u_3, err2_recons=std_rms_u_4,
                     labels1=labels3, labels2=labels4, xlabel='(b) RMS u cases E1 \& E2', colors=colors, DNS=False, z_label=None, z_tics=False)
    #ax2.text(0.5, 0.5, 'To be filled/removed', ha='center', va='center', transform=ax2.transAxes)
    #ax2.invert_yaxis()
    #ax2.set_xlabel('u RMS')

    # Row 2: RMS Vorticity for x and z components
    #ax3 = fig.add_subplot(gs[1, 0])
    #if rms_vort_x_gt is not None and rms_vort_x_recon is not None:
    #    plot_two_curves(ax3, rms_vort_x_gt, rms_vort_x_recon, depth, labels=['Ground Truth', 'Reconstruction'], xlabel='RMS Vorticity (x)')
    #else:
    #    ax3.text(0.5, 0.5, 'Vorticity not computed', ha='center', va='center', transform=ax3.transAxes)
    #    ax3.invert_yaxis()
    #    ax3.set_xlabel('RMS Vorticity (x)')

    #ax4 = fig.add_subplot(gs[1, 1])
    #if rms_vort_z_gt is not None and rms_vort_z_recon is not None:
    #    plot_two_curves(ax4, rms_vort_z_gt, rms_vort_z_recon, depth, labels=['Ground Truth', 'Reconstruction'], xlabel='RMS Vorticity (z)')
    #else:
    #    ax4.text(0.5, 0.5, 'Vorticity not computed', ha='center', va='center', transform=ax4.transAxes)
    #    ax4.invert_yaxis()
    #    ax4.set_xlabel('RMS Vorticity (z)')

    # Row 3: NMSE and PSD Error
    ax5 = fig.add_subplot(gs[1, 0])
    DNS_case1 = 'S1'
    DNS_case2 = 'S2'
    teetank_case1= 'E1'
    teetank_case2= 'E2'
    #plot_single_curve(ax5, nmse_data_1, nmse_data_2, depth1, depth2, err=std_nmse, label1=DNS_case1, label2 =DNS_case2, xlabel='Normalized Mean Square Error', colors=colors, z_label=z_label)
    plot_single_curve(ax5, nmse_data_1, nmse_data_2, nmse_data_3, nmse_data_4, depth1, depth2, depth3, depth4, None, DNS_case1, DNS_case2, teetank_case1, teetank_case2, xlabel='(c) Normalized Mean Squared Error', colors=colors, xlim=True, z_label=z_label)
    ax6 = fig.add_subplot(gs[1, 1])
    plot_single_curve(ax6, psd_data_1, psd_data_2, psd_data_3, psd_data_4, depth1, depth2, depth3, depth4, None, DNS_case1, DNS_case2, teetank_case1, teetank_case2, xlabel='(d) Power Spectral Density Error', colors=colors, xlim=True, z_label=None, z_tics=False)

    # Row 4: SSIM and PSNR
    ax7 = fig.add_subplot(gs[2, 0])
    plot_single_curve(ax7, ssim_data_1, ssim_data_2, ssim_data_3, ssim_data_4, depth1, depth2, depth3, depth4, None, DNS_case1, DNS_case2, teetank_case1, teetank_case2, xlabel='(e) SSIM', colors=colors, xlim=True, z_label=z_label,)

    ax8 = fig.add_subplot(gs[2, 1])
    plot_single_curve(ax8, psnr_data_1, psnr_data_2, psnr_data_3, psnr_data_4, depth1, depth2, depth3, depth4, None, DNS_case1, DNS_case2, teetank_case1, teetank_case2, xlabel='(f) PSNR', colors=colors, z_label=None, z_tics=False)

    # Adjust layout

   
    fig.tight_layout()
    plt.show()
    adr_loc_3 = "C:\\Users\krissmoe\OneDrive - NTNU\PhD\PhD code\PhD-1\Flow Reconstruction and SHRED\Results\\DNS_cases_" 
    filename = adr_loc_3 + "error_metrics"
    filename2 = filename + ".png"
    fig.savefig(filename2)
    fig.savefig(filename+".pdf", format='pdf', bbox_inches='tight', pad_inches=0.1)
    return fig, gs


def plot_time_series_error_metrics(vel_planes_DNS1, vel_planes_DNS2, vel_planes_Tee1, vel_planes_Tee2, SHRED_ensembles, SHRED_ensembles_tee, Tee_ensembles):
    
    RMS_true_RE1000, RMS_recons_RE1000, mse_snapshots_RE1000, ssim_snapshots_RE1000, psnr_snapshots_RE1000, psd_snapshots_RE1000= processdata.calculate_temporal_error_metrics('RE1000', 225, vel_planes_DNS1, 3, SHRED_ensembles, lags=52, forecast=False, add_surface=False, full_planes=True, new_naming=True)
    RMS_true_RE2500, RMS_recons_RE2500,mse_snapshots_RE2500, ssim_snapshots_RE2500, psnr_snapshots_RE2500, psd_snapshots_RE2500= processdata.calculate_temporal_error_metrics('RE2500', 250, vel_planes_DNS2, 3, SHRED_ensembles, lags=52, forecast=False, add_surface=False, full_planes=True, new_naming=True)
    RMS_true_P25, RMS_recons_P25,mse_snapshots_P25, ssim_snapshots_P25, psnr_snapshots_P25, psd_snapshots_P25 = processdata.calculate_temporal_error_metrics_tee('P25', 100, None, vel_planes_Tee1, 3, SHRED_ensembles_tee, Tee_ensembles, lags=52, forecast=False, add_surface=False, full_planes=True, new_naming=True)
    RMS_true_P50, RMS_recons_P50,mse_snapshots_P50, ssim_snapshots_P50, psnr_snapshots_P50, psd_snapshots_P50 = processdata.calculate_temporal_error_metrics_tee('P50', 100, None, vel_planes_Tee2, 3, SHRED_ensembles_tee, Tee_ensembles, lags=52, forecast=False, add_surface=False, full_planes=True, new_naming=True)
    
    test_snaps1=np.arange(0,len(mse_snapshots_RE1000))
    test_snaps2=np.arange(0,len(mse_snapshots_RE2500))
    test_snaps3= np.arange(0, len(mse_snapshots_P25))
    test_snaps4=np.arange(0,len(mse_snapshots_P50))
        
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.2, wspace=0.1)
    

    # Row 1: RMS for u and w
    ax1 = fig.add_subplot(gs[0, 0])
    #ax1.plot(test_snaps1, mse_snapshots_RE1000, label='MSE S1')
    #ax1.plot(test_snaps1, psd_snapshots_RE1000, label='PSD error S1')
    #ax1.plot(test_snaps1, ssim_snapshots_RE1000, label='SSIM S1')
    
    #ax2.plot(test_snaps1, psnr_snapshots_RE1000, color='r', label='PSNR S1')
    #ax2.tick_params(axis='y', labelcolor='r')
    ax1.plot(test_snaps1[200:600], RMS_true_RE1000[200:600], linestyle='-', color='k')
    ax1.plot(test_snaps1[200:600], RMS_recons_RE1000[200:600], linestyle='--', color='blue')
    ax1.set_ylabel('RMS u', fontsize=15)
    ax1.grid()
    ax1.set_ylim(bottom=0.05, top=0.25)
    #ax2 = ax1.twinx()
    #ax2.plot(test_snaps1[200:600], mse_snapshots_RE1000[200:600], linestyle='dotted', label='MSE', color='darkred')
    #ax2.tick_params(axis='y', labelcolor='darkred')
    #ax2.set_ylim(bottom=0.0, top=0.35)
    #ax4.set_y_label('SSIM', fontsize=15)

    ax3 = fig.add_subplot(gs[0, 1])
    #ax3.plot(test_snaps2, mse_snapshots_RE2500, label='MSE S2')
    #ax3.plot(test_snaps2, psd_snapshots_RE2500, label='PSD error S2')
    #ax3.plot(test_snaps2, ssim_snapshots_RE2500, label='SSIM S2')
    #ax4 = ax3.twinx()
    #ax4.plot(test_snaps2, psnr_snapshots_RE2500, color='r', label='PSNR S2')
    #ax4.tick_params(axis='y', labelcolor='r')
    ax3.plot(test_snaps2[200:600], RMS_true_RE2500[200:600], linestyle='-', color='k')
    ax3.plot(test_snaps2[200:600], RMS_recons_RE2500[200:600], linestyle='--', color='blue')
    ax3.set_ylim(bottom=0.05, top=0.25)
    ax3.tick_params(left=False, labelleft=False)
    ax3.grid()
    #ax4 = ax3.twinx()
    #ax4.plot(test_snaps2[200:600], mse_snapshots_RE2500[200:600], linestyle='dotted', label='MSE', color='darkred')
    #ax4.tick_params(axis='y', labelcolor='darkred')
    #ax4.set_ylim(bottom=0.0, top=0.35)
    #ax4.set_ylabel('MSE', fontsize=15)

    ax5 = fig.add_subplot(gs[1,0])
    ax5.plot(test_snaps3, RMS_true_P25, linestyle='-', color='k')
    ax5.plot(test_snaps3, RMS_recons_P25, linestyle='--', color='blue')
    ax5.set_ylim(bottom=0.6, top=5.0)
    ax5.set_ylabel('RMS u', fontsize=15)
    ax5.grid()
    ax5.set_xlabel('test snapshot', fontsize=15)
    #ax6 = ax5.twinx()
    #ax6.plot(test_snaps3, mse_snapshots_P25, linestyle='dotted', label='MSE', color='darkred')
    #ax6.set_ylim(bottom=0.0, top=0.35)

    ax7 = fig.add_subplot(gs[1,1])
    ax7.plot(test_snaps4, RMS_true_P50, linestyle='-', color='k')
    ax7.plot(test_snaps4, RMS_recons_P50, linestyle='--', color='blue')
    ax7.set_ylim(bottom=0.6, top=5.0)
    ax7.grid()
    ax7.set_xlabel('test snapshot', fontsize=15)
    ax7.tick_params(left=False, labelleft=False)    
    #ax8 = ax7.twinx()
    #ax8.plot(test_snaps4, mse_snapshots_P50, linestyle='dotted', label='SSIM', color='darkred')
    #ax8.tick_params(axis='y', labelcolor='darkred')
    #ax8.set_ylim(bottom=0.0, top=0.35)
    #ax8.set_ylabel('MSE', fontsize=15)
    for ax in [ax1, ax3, ax5, ax7]:
        ax.tick_params(axis='x', which='major', labelsize=13)

    # Apply to y-axis of ax1 and ax5
    ax1.tick_params(axis='y', which='major', labelsize=13)
    ax5.tick_params(axis='y', which='major', labelsize=13)  
    
    fig.tight_layout()
    plt.show()
    adr_loc_3 = "C:\\Users\krissmoe\OneDrive - NTNU\PhD\PhD code\PhD-1\Flow Reconstruction and SHRED\Results\\time_series_error_metrics" 
    filename = adr_loc_3
    filename2 = filename + ".png"
    fig.savefig(filename2)
    fig.savefig(filename+".pdf", format='pdf', bbox_inches='tight', pad_inches=0.1) 

    
def plot_instantaneous_RMS(experimental_ens,  SHRED_ens_DNS, SHRED_ens_exp, snap_indices_DNS, snap_indices_exp, ranks, num_sensors, colors, labels, z_norm='int'):
    """
    Visualise *instantaneous* root-mean-square (RMS) vertical profiles for
    four flow cases – two DNS (S1/S2) and two experimental (E1/E2) – and
    compare ground truth against SHRED reconstructions.

    A 4 × 3 panel figure is created:

    ==========================  =========================================
    Row                          Cases plotted (left → right)
    --------------------------  -----------------------------------------
    0 (top)                     S1 (RE1000) snapshots *i = 0,1,2*
    1                            S2 (RE2500) snapshots *i = 0,1,2*
    2                           E1 (P25)    snapshots *i = 0,1,2*
    3 (bottom)                  E2 (P50)    snapshots *i = 0,1,2*
    ==========================  =========================================

    Each subplot shows:

    * Solid line – ground-truth RMS profile at snapshot *i*
    * Dashed line – SHRED-reconstructed RMS profile
    * Depth axis optionally normalised (``z_norm``)

    The final figure is saved both as PNG and PDF.

    Parameters
    ----------
    experimental_ens : int
        Experimental ensemble index used to load SVD/RMS data.
    SHRED_ens_DNS : int
        SHRED ensemble seed for DNS cases.
    SHRED_ens_exp : int
        SHRED ensemble seed for experimental cases.
    snap_indices_DNS : list[int]
        Three snapshot indices to plot for S1 and S2.
    snap_indices_exp : list[int]
        Three snapshot indices to plot for E1 and E2.
    ranks : tuple[int, int, int, int]
        SVD truncation ranks (unused here but reserved for future use).
    num_sensors : int
        Number of surface sensors (used only in filename bookkeeping).
    colors : tuple[str, str, str, str]
        Line colours for S1, S2, E1, E2 (in that order).
    labels : tuple[tuple[str, str], ...]
        Label pairs ``(ground_truth_label, recon_label)`` per snapshot.
    z_norm : {"int", "taylor", "mixed", None}, default "int"
        Normalisation for the vertical axis:
        * ``"int"``  →  integral length scale  
        * ``"taylor"`` → Taylor microscale  
        * ``"mixed"``  →  √(Lλ L∞)  
        * ``None``     →  raw grid index

    Returns
    -------
    None
        Produces a Matplotlib figure and writes files:

        * ``.../Results/DNS_cases_rms_instantaneous.png``
        * ``.../Results/DNS_cases_rms_instantaneous.pdf``

    Notes
    -----
    * Requires pre-computed `.mat` files created by
      ``processdata.open_instantaneous_rms_profile(_exp)``.
    """

    rms_gt_S1, rms_recons_S1 = processdata.open_instantaneous_rms_profile('RE1000', SHRED_ens_DNS)
    rms_gt_S2, rms_recons_S2 = processdata.open_instantaneous_rms_profile('RE2500', SHRED_ens_DNS)
    rms_gt_E1, rms_recons_E1 = processdata.open_instantaneous_rms_profile_exp('P25', experimental_ens, SHRED_ens_exp)
    rms_gt_E2, rms_recons_E2 = processdata.open_instantaneous_rms_profile_exp('P50', experimental_ens, SHRED_ens_exp)
    
    #choose snap indices
    rms_gt_S1 = rms_gt_S1[:, snap_indices_DNS]
    rms_recons_S1 = rms_recons_S1[:,snap_indices_DNS]

    rms_gt_S2= rms_gt_S2[:,snap_indices_DNS]
    rms_recons_S2 = rms_recons_S2[:,snap_indices_DNS]

    rms_gt_E1 = rms_gt_E1[:, snap_indices_exp]
    rms_recons_E1 = rms_recons_E1[:,snap_indices_exp]

    

    rms_gt_E2= rms_gt_E2[:,snap_indices_exp]
    rms_recons_E2 = rms_recons_E2[:,snap_indices_exp]
    
    #plot
    z_S1 = utilities.get_zz_DNS('RE1000')
    z_S2 = utilities.get_zz_DNS('RE2500')
    z_S1 = utilities.get_normalized_z(z_S1, z_norm, 'RE1000')
    z_S2 = utilities.get_normalized_z(z_S2, z_norm, 'RE2500')

    z_exp = utilities.get_zz_exp()
    z_E1 = utilities.get_normalized_z_exp(z_exp, z_norm, exp_case='P25')
    z_E2 = utilities.get_normalized_z_exp(z_exp, z_norm, exp_case='P50')

    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
    })
    
    fig = plt.figure(figsize=(14, 15))
    gs = GridSpec(4, 3, figure=fig, hspace=0.2, wspace=0.1)
    


    labels1=['Ground Truth, Re=1000', 'Recons, Re=1000']
    #make z_label:
    if z_norm=='taylor':
        z_label='$z/L_{\lambda}$'
    elif z_norm == 'int':
        z_label='$z/L_{\infty}$'
    elif z_norm=='mixed':
        z_label='$z/(L_{\lambda} L_{\infty})^{1/2}$'  
    else:
        z_label = None
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    ax4 = fig.add_subplot(gs[1, 0])
    ax5= fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    ax7 = fig.add_subplot(gs[2, 0])
    ax8= fig.add_subplot(gs[2, 1])
    ax9 = fig.add_subplot(gs[2, 2])

    ax10 = fig.add_subplot(gs[3, 0])
    ax11= fig.add_subplot(gs[3, 1])
    ax12 = fig.add_subplot(gs[3, 2])
    
    axes = [[ax1, ax2, ax3], [ax4, ax5, ax6], [ax7, ax8, ax9], [ax10, ax11, ax12]]

    for i in range(len(snap_indices_DNS)):
        #Row 1: RMS for u and w
        print(i)
        ax_S1 = axes[0][i]
        ax_S1.plot(rms_recons_S1[:,i], z_S1, linestyle='--', color=colors[0], label=labels[i][1])
        ax_S1.plot(rms_gt_S1[:,i], z_S1, linestyle='-', color=colors[0], label=labels[i][0])

        ax_S2 = axes[1][i]
        ax_S2.plot(rms_recons_S2[:,i], z_S2, linestyle='--', color=colors[1], label=labels[i][1])
        ax_S2.plot(rms_gt_S2[:,i], z_S2, linestyle='-', color=colors[1], label=labels[i][0])

        ax_E1 = axes[2][i]
        ax_E1.plot(rms_recons_E1[:,i], z_E1[1:], marker='x', linestyle='--', color=colors[2], label=labels[i][1])
        ax_E1.plot(rms_gt_E1[:,i], z_E1[1:], marker='o', linestyle='-',  color=colors[2], label=labels[i][0])

        ax_E2 = axes[3][i]
        ax_E2.plot(rms_recons_E2[:,i], z_E2[1:], marker='x', linestyle='--', color=colors[3], label=labels[i][1])
        ax_E2.plot(rms_gt_E2[:,i], z_E2[1:], marker='o', linestyle='-', color=colors[3], label=labels[i][0])
        
        ax_S1.set_xlabel('RMS u, S1', fontsize=14)
        ax_S2.set_xlabel('RMS u, S2', fontsize=14)
        ax_E1.set_xlabel('RMS u, E1', fontsize=14)
        ax_E2.set_xlabel('RMS u, E2', fontsize=14)

        ax_S1.set_xlim((0.06,0.21))
        ax_S2.set_xlim((0.055,0.19))
        ax_E1.set_xlim((0.65,2.3))
        ax_E2.set_xlim((0.3,2.8))
        if i>0:
            ax_S1.set_yticklabels([])
            ax_S2.set_yticklabels([])
            ax_E1.set_yticklabels([])
            ax_E2.set_yticklabels([])
        ax_S1.grid()
        ax_S2.grid()
        ax_E1.grid()
        ax_E2.grid()
        if i==0:
            ax_S1.legend(fontsize=13)
            ax_S2.legend(fontsize=13)
            ax_E1.legend(fontsize=13)
            ax_E2.legend(fontsize=13)

    ax1.set_ylabel(z_label, fontsize=16)
    ax4.set_ylabel(z_label, fontsize=16)
    ax7.set_ylabel(z_label, fontsize=16)
    ax10.set_ylabel(z_label, fontsize=16)
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]:
        ax.tick_params(axis='x', which='major', labelsize=13)
        if ax==ax1 or ax==ax4 or ax==ax7 or ax==ax10:
            ax.tick_params(axis='y', which='major', labelsize=13)
    fig.tight_layout()
    plt.show()
    adr_loc_3 = "C:\\Users\krissmoe\OneDrive - NTNU\PhD\PhD code\PhD-1\Flow Reconstruction and SHRED\Results\\DNS_cases_" 
    filename = adr_loc_3 + "rms_instantaneous"
    filename2 = filename + ".png"
    fig.savefig(filename2)
    fig.savefig(filename+".pdf", format='pdf', bbox_inches='tight', pad_inches=0.1) 




def plot_SHRED_comparison_DNS(r_new, SHRED_ens, vel_planes, num_sensors, test_snap_index, lags=52, forecast=False, add_surface=False, full_planes=True, DNS_case='S2'):
    """
    Create side-by-side image comparisons of ground truth, SVD truncation,
    and SHRED reconstruction for selected DNS planes (and optionally the
    free surface), then save the figures to disk. If more than 2 velocity planes, 
    the comparisons are plotted one by one. Else, we combine them in one figure.

    Parameters
    ----------
    r_new : int
        Truncation rank used for both SVD and SHRED reconstruction.
    SHRED_ens : int
        SHRED ensemble index (seed) used when loading saved `.mat` files.
    vel_planes : list[int]
        DNS velocity-plane indices to visualise.
    num_sensors : int
        Number of surface sensors present in the SHRED input (used only
        for filename construction).
    test_snap_index : int
        Time index of the snapshot to display from the test set.
    lags : int, default 52
        LSTM sequence length used when SHRED was trained (needed for
        correct indexing in the test set).
    forecast : bool, default False
        If *True*, load forecast-mode SHRED results; otherwise reconstruction.
    add_surface : bool, default False
        If *True*, plot an additional figure for the surface elevation.
    full_planes : bool, default True
        If *True*, SHRED was trained on **all** planes; otherwise only the
        subset in ``vel_planes`` (affects plane indexing).
    DNS_case : {"S1", "S2"}, default "S2"
        DNS dataset identifier.

    Returns
    -------
    None
        Generates and saves PNG figures; no data are returned.

    Notes
    -----
    * Assumes SHRED outputs are stored in `.mat` files located in
      `adr_loc_3` with naming convention used in
      :pyfunc:`utilities.open_SHRED`.
    * Figure filenames follow
      ``compare_recon_<DNS_case>_plane<plane>_rank<r>_DNS_ens<SHRED_ens>.png``..
    """
    
    DNS_case = utilities.case_name_converter(DNS_case)

    #set location folder for plots
    #TODO: change address
    adr_loc_3 = "C:\\Users\krissmoe\OneDrive - NTNU\PhD\PhD code\PhD-1\Flow Reconstruction and SHRED\Results\\DNS"
    
    if DNS_case=='RE2500':
        tot_num_planes=76
    else:
        tot_num_planes=57

    #create mesh
    XX,YY = utilities.get_mesh_DNS(DNS_case)

    #get SHRED V matrices
    V_test_recons, V_test_ground_truth, V_test_indices = utilities.open_SHRED(None, DNS_case, r_new, num_sensors, SHRED_ens, vel_planes, DNS=True, full_planes=full_planes, forecast=forecast)

    #get surface elevation
    all_rows = []
    
    if add_surface:
        plane=0 #correct plane number for surface elevation 
        plane_index=0
        surf_fluc_test, surf_svd_test, surf_recons_test, u_fluc2 = utilities.get_test_imgs_SHRED_DNS(DNS_case, plane, plane_index, np.zeros(1), V_test_recons, V_test_indices, r_new,
                                                                                                num_sensors, lags=lags, forecast=forecast, surface=True, no_input_u_fluc=True)
        del u_fluc2

        #extract specific snapshot
        surf_fluc_test = surf_fluc_test[:, :, test_snap_index]
        surf_svd = surf_svd_test[:, :, test_snap_index]
        surf_recons = surf_recons_test[:, :, test_snap_index]
        row1 = [surf_fluc_test, surf_svd, surf_recons]  # First row
        all_rows.append(row1)
        
        if len(vel_planes) >= 3:
            spacing = 0.1
            fig, axs = plt.subplots(1, 3, figsize=(15, 5),  gridspec_kw={'wspace': spacing, 'hspace': spacing})
            for j, snapshot in enumerate(row1):
                ax = axs[j]
                if j==0:
                    min_val = np.min(snapshot)
                    max_val = np.max(snapshot)
            
                ax.imshow(snapshot,  cmap=cmocean.cm.ice, interpolation='bilinear', vmin=min_val, vmax=max_val)
                ax.axis('off')

            filename = adr_loc_3 + "compare_recon_SURF_ELEV_"+DNS_case+"_rank"+str(r_new)+ "_DNS_ens" + str(SHRED_ens) +".png"
            plt.savefig(filename, format='png', bbox_inches='tight', pad_inches=0.5)
        
            plt.show()
    
    #iterate and plot velocity planes
    for i in range(len(vel_planes)):
        print("plane ", vel_planes[i])
        plane = vel_planes[i]
        if full_planes and len(vel_planes) < tot_num_planes:
            plane_index = vel_planes[i]
        else:
            plane_index = i+1 #shift with 1 to compensate for surface elevation when loading V matrices
        #construct raw fields
        u_fluc = utilities.get_velocity_plane_DNS(DNS_case, plane)
        
        print("getting test images")
        u_fluc_test, u_svd_test, u_recons_test, u_fluc2 = utilities.get_test_imgs_SHRED_DNS(DNS_case, plane, plane_index, u_fluc, V_test_recons, V_test_indices, r_new,
                                                                                            num_sensors, lags=lags, forecast=forecast, surface=False, no_input_u_fluc=True)
        del u_fluc2
        print("test images extracted")
        #construct SVD fields
        u_fluc_test = u_fluc_test[:,:,test_snap_index]
        u_svd = u_svd_test[:, :, test_snap_index]

        #construct SHRED reconstruction fields
        
        u_recons = u_recons_test[:, :, test_snap_index]

        #all_rows_1 = []
        row = [u_fluc_test, u_svd, u_recons]  # Second row
        #all_rows_1.append(row)

        if len(vel_planes) < 3:
            all_rows.append(row)

        # Plot each snapshot in the grid
        if len(vel_planes) >= 3:
        
            spacing = 0.1
            fig, axs = plt.subplots(1, 3, figsize=(15, 5),  gridspec_kw={'wspace': spacing, 'hspace': spacing})
        
            for j, snapshot in enumerate(row):
                #colour_levels = np.linspace(-u_scale,u_scale, 500)
                ax = axs[j]
                if j==0:
                    min_val = np.min(snapshot)
                    max_val = np.max(snapshot)
                ax.imshow(snapshot,  cmap='RdBu_r', interpolation='bilinear', vmin=min_val, vmax=max_val)
                ax.axis('off')

            filename = adr_loc_3 + "compare_recon_"+DNS_case+"_plane" + str(plane) + "_rank"+str(r_new)+ "_DNS_ens" + str(SHRED_ens) +".png"
            plt.savefig(filename, format='png', bbox_inches='tight', pad_inches=0.5)
        
            plt.show()
            plt.close()



    #plot 3x3 subplots with surface elevation, surface velocity and bulk velocity
    #plot for paper
    if len(vel_planes) < 3:
        spacing = 0.1
        fig, axs = plt.subplots(len(all_rows), 3, figsize=(15, 5*len(all_rows)),  gridspec_kw={'wspace': spacing, 'hspace': spacing})
        cmaps = [cmocean.cm.ice,'RdBu_r','RdBu_r'] # Alternative colors for velocity: cm.thermal, cm. balance
        ax1 = axs[0,0]
        ax2 = axs[0,1]
        ax3 = axs[0,2]
        ax1.set_title("Ground truth")
        ax2.set_title("Compressed")
        ax3.set_title("Reconstruction")
        for k, row in enumerate(all_rows):
            for j, snapshot in enumerate(row):
                if j==0:
                    min_val = np.min(snapshot)
                    max_val = np.max(snapshot)

                ax = axs[k,j]
             
                ax.imshow(snapshot,  cmap=cmaps[k], interpolation='bilinear', vmin=min_val, vmax=max_val)
                
                ax.axis('off')

    plt.show()
    filename = adr_loc_3 + "compare_recon_"+DNS_case+"_rank"+str(r_new)+ "_" + "DNS_ens" + str(SHRED_ens) + "_" + str(test_snap_index) +".png"
    fig.savefig(adr_loc_3 + "compare_recon_DNS_"+DNS_case+"_rank"+str(r_new)+ "_DNS_ens" + str(SHRED_ens) + "_" + str(test_snap_index) + ".pdf", format='pdf', bbox_inches='tight', pad_inches=0.5, transparent=False, dpi=150)
    plt.close()



def plot_SHRED_comparison_exp(r_new, exp_case, experimental_ens, SHRED_ens, plane_list, test_snap_index, u_fluc=None, surf_fluc=None, 
                              num_sensors=3, lags=52, forecast=False, add_surface=False):
    """
    Plot ground-truth, rank-r SVD, and SHRED-reconstructed snapshots of
    T-tank experimental data for the chosen velocity planes (and
    optionally the surface), then save the figures.


    Parameters
    ----------
    r_new : int
        SVD truncation rank used for SHRED and comparison plots.
    exp_case : {"E1", "E2"}
        Experimental case identifier (converted internally to "E1"/"E2").
    experimental_ens : int
        Experimental ensemble index to read SVD data from.
    SHRED_ens : int
        SHRED ensemble seed used when saving reconstructions.
    plane_list : list[int]
        Indices (1-based: 1 = H395, 2 = H390, …) of velocity planes to plot.
    test_snap_index : int
        Snapshot index within the test set to visualise.
    u_fluc, surf_fluc : ndarray or None
        Optional pre-loaded velocity / surface time series.  If *None*,
        data are loaded internally.
    num_sensors : int, default 3
        Number of surface sensors used in SHRED (filename bookkeeping).
    lags : int, default 52
        LSTM sequence length used in training; required for correct
        indexing when extracting test snapshots.
    forecast : bool, default False
        If *True*, load forecast-mode SHRED reconstructions.
    add_surface : bool, default False
        Plot an additional figure for surface elevation.

    Returns
    -------
    None

    Notes
    -----
    * Assumes SHRED outputs are stored in `.mat` files and accessed via
      :pyfunc:`utilities.open_SHRED`.
    * Hard-coded output path `adr_loc_3`; adjust for other environments.
    """
    #TODO: Add other velocity fields to the plotting function as I get them
    #set location folder for plots
    
    exp_case = utilities.case_name_converter(exp_case)
    adr_loc_3 = "C:\\Users\krissmoe\OneDrive - NTNU\PhD\PhD code\PhD-1\Flow Reconstruction and SHRED\Results\Tee_tank"
    
    
    #iterate test snaps:
    #filenames_snaps = []
  
    
    


    for i in range(len(plane_list)):
        
        planes = ['H395', 'H390', 'H375', 'H350', 'H300']
        plane = planes[plane_list[i]-1]
        print("plane: ", plane)
        X_surf, Y_surf, X_vel, Y_vel = utilities.get_mesh_exp(exp_case, plane)
        #plane_index = i+1
        #construct raw fields

        #get SHRED V matrices
        test_recons, test_ground_truth, test_indices = utilities.open_SHRED(experimental_ens, exp_case, r_new, num_sensors, 
                                                                            SHRED_ens, plane_list, DNS=False, plane=plane, full_planes=True, forecast=forecast)
        

        #get surface elevation
        all_rows = []
        if add_surface:
            #plane=0
            #plane_index=0
            surf_fluc_test, surf_svd_test, surf_recons_test, u_fluc2 = utilities.get_test_imgs_SHRED_exp(plane, surf_fluc, u_fluc, test_recons, test_ground_truth, test_indices,
                                                                                                            X_surf, X_vel, experimental_ens, exp_case, r_new, 
                                                                                                            SHRED_ens, num_sensors, U_tot_red=None, S_tot_red=None, V_tot_red = None, open_svd=True,
                                                                                                              lags=lags, forecast=forecast, surface=True, no_input_u_fluc=True)
            del u_fluc2
            surf_fluc_test = surf_fluc_test[:, :, test_snap_index]
            surf_svd = surf_svd_test[:, :, test_snap_index]
            surf_recons = surf_recons_test[:, :, test_snap_index]
            row1 = [surf_fluc_test, surf_svd, surf_recons]  # First row
            all_rows.append(row1)
            if len(plane_list) >= 3:
                spacing = 0.1
                fig, axs = plt.subplots(1, 3, figsize=(15, 5),  gridspec_kw={'wspace': spacing, 'hspace': spacing})
                for j, snapshot in enumerate(row1):
                    if j==0:
                        min_val = np.min(snapshot)
                        max_val = np.max(snapshot)
              
                    ax = axs[j]
                    ax.imshow(snapshot, cmap=cmaps[k], interpolation='bilinear', vmin=min_val, vmax=max_val)
                    ax.axis('off')

                filename = adr_loc_3 + "Teetank_compare_recon_SURF_ELEV_rank"+str(r_new)+ "_" +exp_case +"_"+plane +  "_tee_ens_"+str(experimental_ens)+"_SHREDens" + str(SHRED_ens) +".png"
                plt.savefig(filename, format='png', bbox_inches='tight', pad_inches=0.5)
        
                plt.show()
        



        print("getting test images")
        u_fluc_test, u_svd_test, u_recons_test, u_fluc2 = utilities.get_test_imgs_SHRED_exp(plane, surf_fluc, u_fluc, test_recons, test_ground_truth, test_indices, X_surf, X_vel, experimental_ens, exp_case, r_new, 
                                                            SHRED_ens, num_sensors, U_tot_red=None, S_tot_red=None, V_tot_red = None, open_svd=True, lags=lags, forecast=forecast, surface=False, no_input_u_fluc=True)
        del u_fluc2
        print("test images extracted")
        #construct SVD fields
        u_fluc_test = u_fluc_test[:,:,test_snap_index]
        u_svd = u_svd_test[:, :, test_snap_index]

        #construct SHRED reconstruction fields
        
        u_recons = u_recons_test[:, :, test_snap_index]

        #all_rows_1 = []
        row = [u_fluc_test, u_svd, u_recons]  # Second row
        #all_rows_1.append(row)

        if len(plane_list) < 3:
            all_rows.append(row)

        # Plot each snapshot in the grid
        if len(plane_list) >= 3:
            spacing = 0.1
            fig, axs = plt.subplots(1, 3, figsize=(15, 5),  gridspec_kw={'wspace': spacing, 'hspace': spacing})
        
            for j, snapshot in enumerate(row):
                if j==0:
                    min_val = np.min(snapshot)
                    max_val = np.max(snapshot)
            
        
                ax = axs[j]
                ax.imshow(snapshot, cmap=cmaps[k], interpolation='bilinear', vmin=min_val, vmax=max_val)
                ax.axis('off')


        # Adjust layout and display the plot
        # plt.tight_layout()
        #plt.show()

            filename = adr_loc_3 + "Teetank_compare_recon_u_rank"+str(r_new)+ "_" +exp_case +"_"+plane +  "_tee_ens_"+str(experimental_ens)+"_SHREDens" + str(SHRED_ens) +".png"
            plt.savefig(filename, format='png', bbox_inches='tight', pad_inches=0.5)
        
            plt.show()
            plt.close()



    #plot 3x3 subplots with surface elevation, surface velocity and bulk velocity
    #plot for paper
    if len(plane_list) < 3:
        spacing = 0.1
        fig, axs = plt.subplots(len(all_rows), 3, figsize=(15, 5*len(all_rows)),  gridspec_kw={'wspace': spacing, 'hspace': spacing})
        cmaps = [cmocean.cm.ice,'RdBu_r','RdBu_r'] # Alternative colors for velocity: cm.thermal, cm. balance
        
        for k, row in enumerate(all_rows):
            for j, snapshot in enumerate(row):
                if j==0:
                    min_val = np.min(snapshot)
                    max_val = np.max(snapshot)



                ax = axs[k,j]
                #contour1 = ax.contourf(XX,YY, snapshot, levels = colour_levels, cmap = cmaps[k])
                ax.imshow(snapshot, cmap=cmaps[k], interpolation='bilinear', vmin=min_val, vmax=max_val)
                ax.axis('off')
                #ax.axis('square')
                #my_ticks = np.arange(-0.75,0.75,0.15)
                #cb = plt.colorbar(ticks=my_ticks)
                #ax1.tick_params(axis='both', labelsize=16)
                #cb.set_label(label='u [m/s]', fontsize=16)
                #fig.colorbar(contour1, ax=ax1, ticks=my_ticks, label="u' [m/s]")


                #ax = axs[k, j]
                #ax.imshow(snapshot, cmap=cmaps[k], interpolation='bilinear', aspect='auto')
                #ax.axis('off')
        plt.show()
        filename = adr_loc_3 + "Teetank_compare_recon_TOTAL_rank"+str(r_new)+ "_" +exp_case +"_"+plane +  "_tee_ens_"+str(experimental_ens)+"_SHREDens" + str(SHRED_ens)
        #plt.savefig(filename + ".png", format='png', bbox_inches='tight', pad_inches=0.5)
        fig.savefig(filename + ".eps", format='eps', bbox_inches='tight', pad_inches=0.5)
      

def loop_SHRED_comparison_DNS(r_new, SHRED_ens, vel_planes, num_sensors,test_index_start, test_index_end, surf_scale, u_scale, add_surface=False, full_planes=True, DNS_case='RE2500'):
    
    indices = np.arange(test_index_start, test_index_end+1)
    
    for i in range(len(indices)):
        test_snap_index = indices[i]
        plot_SHRED_comparison_DNS(r_new, SHRED_ens, vel_planes, num_sensors, test_snap_index, surf_scale, u_scale, lags=52, forecast=False, add_surface=add_surface, full_planes=True, DNS_case=DNS_case)


def make_GIF_comparison_DNS(r_new, SHRED_ens, vel_planes, num_sensors,test_index_start, test_index_end, surf_scale, u_scale, add_surface=False, full_planes=True, DNS_case='RE2500', gif_name='DNS GIF'):

    indices = np.arange(test_index_start, test_index_end+1)
    filenames = []
    adr_loc_3 = "C:\\Users\krissmoe\OneDrive - NTNU\PhD\PhD code\PhD-1\Flow Reconstruction and SHRED\Results\\"

    for i in range(len(indices)):
        test_snap_index = indices[i]
        fname = adr_loc_3 + "DNScompare_recon_DNS_"+DNS_case+"_rank"+str(r_new)+ "_" + "DNS_ens" + str(SHRED_ens) + "_" + str(test_snap_index) +".png"
        filenames.append(fname) 
    utilities.create_GIF(filenames, gif_name)

def plot_parameter_analysis_DNS(DNS_case, r_vals, vel_planes, sensor_vals, optimal_var_val, SHRED_ensembles, full_planes=False, r_analysis=True, singular_val_energy=False, comp_rate=False):
    '''function to plot parameter analysis, where depth and ensemble averaged error metrics are plotted and normalized for a range of variable values.
        Either varying rank r (given by a list r_vals) or varying number of sensors (given by sensor_vals)'''
        
        
    mse_list, ssim_list, psnr_list, psd_list = processdata.calc_avg_error(DNS_case, r_vals, vel_planes, sensor_vals, SHRED_ensembles, forecast=False, full_planes=full_planes,r_analysis=r_analysis)
        
    if r_analysis:
        var_vals = r_vals
        if DNS_case=='RE2500':
            total_ranks = 12500
        else:
            total_ranks=10900
        s_energy, r_percentage = utilities.get_cumsum_svd(r_vals, total_ranks, DNS_case)
        var_str = 'rank'
        if singular_val_energy:
            var_vals = s_energy*100
            var_str = "% SVD cumulative energy"
        if comp_rate:
            var_vals=r_percentage*100
            var_str = "% Of total SVD ranks"
    else:
        var_vals = sensor_vals
        var_str = 'sensors'
        
    rank = r_vals[0]
    plt.plot([optimal_var_val,optimal_var_val],[0.3,1.01], "--", color='k')
    plt.plot(var_vals, psnr_list/np.amax(psnr_list), label="PSNR")
    plt.plot(var_vals, mse_list/np.amax(mse_list), label='MSE')
    plt.plot(var_vals, ssim_list/np.amax(ssim_list), label='SSIM')
    plt.plot(var_vals, psd_list/np.amax(psd_list), label='PSD')
    plt.grid()
    plt.xlabel(var_str,fontsize='16')
    #plt.ylabel('Normalized metric')
    plt.legend(fontsize=13)
    plt.tick_params(axis='x', which='major', labelsize=13)
    plt.tick_params(axis='y', which='major', labelsize=13)
    #plt.ylabel('PSNR')
    plt.savefig(var_str + "_analysis_DNS_"+DNS_case+".pdf")
    plt.show()


def plot_parameter_analysis_tee(Teetank_case, Tee_ensembles, r_vals, vel_planes, sensor_vals, optimal_var_val, SHRED_ensembles, full_planes=False, r_analysis=True, singular_val_energy=False, comp_rate=False):
    '''function to plot parameter analysis, where depth and ensemble averaged error metrics are plotted and normalized for a range of variable values.
        Either varying rank r (given by a list r_vals) or varying number of sensors (given by sensor_vals)'''
        
    mse_list, ssim_list, psnr_list, psd_list =processdata.calc_avg_error_tee(Teetank_case, r_vals, vel_planes, sensor_vals, SHRED_ensembles, Tee_ensembles, forecast=False, add_surface=False, full_planes=full_planes, new_naming=True, r_analysis=r_analysis)
        
    if r_analysis:
        var_vals = r_vals
        total_ranks=900

        s_energy, r_percentage = utilities.get_cumsum_svd_tee(r_vals, total_ranks, Teetank_case, Tee_ensembles[0], plane=2)
        var_str = 'rank'
        if singular_val_energy:
            var_vals = s_energy*100
            var_str = "% SVD cumulative energy"
        if comp_rate:
            var_vals=r_percentage*100
            var_str = "% Of total SVD ranks"
    else:
        var_vals = sensor_vals
        var_str = 'sensors'
        
    rank = r_vals[0]
    plt.plot([optimal_var_val,optimal_var_val],[0.21,1.01], "--", color='k')
    plt.plot(var_vals, psnr_list/np.amax(psnr_list), label="PSNR")
    plt.plot(var_vals, mse_list/np.amax(mse_list), label='MSE')
    plt.plot(var_vals, ssim_list/np.amax(ssim_list), label='SSIM')
    plt.plot(var_vals, psd_list/np.amax(psd_list), label='PSD')
    plt.grid()
    plt.xlabel(var_str,fontsize='16')
    plt.legend(fontsize=13)
    plt.tick_params(axis='x', which='major', labelsize=13)
    plt.tick_params(axis='y', which='major', labelsize=13)
    #plt.ylabel('PSNR')
    plt.savefig(var_str + "_analysis_teetank_"+Teetank_case+".pdf")
    plt.show()


def plot_data_snaps(snap_start, snap_end, data_scale, DNS_case, velocity_plane, surface=False, GIF_only=True):
    if surface:
        data = utilities.get_surface(DNS_case)
        color=cmocean.cm.ice
        plane_str = 'surf'
    else:
        data= utilities.get_velocity_plane_DNS(DNS_case, velocity_plane)
        color= 'RdBu_r'
        plane_str = 'plane' + str(velocity_plane)
    XX, YY = utilities.get_mesh_DNS(DNS_case)
    num_snaps = snap_end-snap_start
    
    #iterate and plot each data snapshot in the interval
    filenames_snaps = []
    for i in range(snap_start, snap_end+1):
        print("snap: ", i)
        snap = data[:,:,i]

        fig, (ax1) = plt.subplots(1, 1, figsize=(5, 5))  # 1 row, 2 columns
        ax1.set_title(str(i))    
        colour_levels = np.linspace(-data_scale,data_scale, 500)
        contour = ax1.contourf(XX,YY, snap, levels = colour_levels, cmap = color)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.legend()
      
        ax1.tick_params(axis='both', labelsize=16)
        #cb.set_label(label='u [m/s]', fontsize=16)
        #fig.colorbar(contour, ax=ax1, ticks=my_ticks, label="$u$ [m/s]")

        # Adjust the layout so the plots do not overlap
        plt.tight_layout()
        if GIF_only:
            adr = 'E:\\Users\krissmoe\Documents\PhD data storage\DNS plot throwaway'
        else:
            adr = 'E:\\Users\krissmoe\Documents\PhD data storage\DNS plot results'
        savestring = adr + DNS_case + str(i) +".png"
        plt.savefig(savestring)
        filenames_snaps.append(savestring)
        #plt.show()

    gif_name = 'C:\\Users\krissmoe\OneDrive - NTNU\PhD\PhD code\PhD-1\Flow Reconstruction and SHRED\Results'+"\\_" + DNS_case + "_"+ plane_str+ "_"+str(snap_start)+"_to" + str(snap_end)
    utilities.create_GIF(filenames_snaps, gif_name)