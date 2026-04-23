import scipy.integrate
import torch
from scipy.io import loadmat
import numpy as np
import utilities
import h5py
from lvpyio import read_set
from sklearn.preprocessing import MinMaxScaler
import models
import torch
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.fft import fft2, fftshift
import scipy.signal as sig
from pathlib import Path
import os
from paths import DNS_RAW_DIR, EXP_RAW_DIR, DNS_SVD_DIR, EXP_SVD_DIR, SHRED_DIR, METRICS_DIR


'''analysis of r dependence of PSD spectra'''

def calculate_PSD_r_vals_DNS(DNS_case, u_fluc, rank_list, num_ens, plane):
    """
    Compute ensemble‑averaged 1‑D power‑spectral densities (PSD) for multiple
    SVD truncation ranks in a selected DNS velocity plane.

    Workflow
    --------
    1.  Load the full‑rank fluctuating velocity field, transpose to
        `(nt, ny, nx)`, and compute a reference PSD.
    2.  Loop over each rank in ``rank_list``  
        a. Load reduced‑rank SVD factors via
           ``utilities.open_and_reduce_SVD``.  
        b. Reconstruct the velocity field at rank ``rank``.  
        c. Compute its PSD and store it for each ensemble.  
    3.  Average PSDs across `num_ens` ensembles for each rank.
    4.  Normalize all PSD curves by the global maximum for plotting.
    5.  Return normalized PSD matrix and the wavenumber axis scaled by the
        integral length scale of the DNS case.

    Parameters
    ----------
    DNS_case : str
        Identifier for the DNS dataset (e.g. "RE1000", "RE2500").
    u_fluc : ndarray
        Full‑rank fluctuating velocity field of shape `(ny, nx, nt)`.
    rank_list : list[int]
        List of truncation ranks to evaluate.  The first element should
        match the baseline `r=1000` used when loading SVDs.
    num_ens : int
        Number of ensemble realizations to average over.
    plane : int
        Velocity‑plane index (0‑based or 1‑based depending on your utility).

    Returns
    -------
    psd_multi : ndarray, shape (len(rank_list), len(k_vals))
        PSD curves normalized to their global maximum.
    k_vals : ndarray
        Non‑dimensional wavenumber bins (scaled by the integral length
        scale of the DNS case).

    Notes
    -----
    * Spatial resolution is inferred from the DNS grid (`dx = dy = 2π/N`).
    * Integrates with `calculate_psd_1d` for PSD calculation.
    * The first row of `psd_multi` corresponds to the full‑rank PSD.
    """

    dimX, dimY, dimT = utilities.get_dims_DNS(DNS_case)
    
    #set spatial resolution based on DNS data
    dx = 2*np.pi/dimX
    dy=2*np.pi/dimY 
    
    u_fluc = np.transpose(u_fluc, (2,0,1))
    PSD_vals, k_vals = calculate_psd_1d(u_fluc, dx=dx, dy=dy, DNS=True)

    integral_length_scale=utilities.get_integral_length_scale(DNS_case)
    
    PSD_vals_all = np.zeros((num_ens, len(PSD_vals)))
    psd_multi_recon = np.zeros((len(rank_list), len(k_vals)))
    psd_multi_recon[0] = PSD_vals
    for i in range(1,len(rank_list)):
        rank = rank_list[i]
       
        PSD_vals_all = np.zeros((num_ens, len(PSD_vals)))
        for j in range(num_ens):

            U, S, V = utilities.open_and_reduce_SVD(None, None, rank, forecast=False, DNS=True, DNS_plane=plane, DNS_surf=False, DNS_case=DNS_case, exp=False, plane=None)
            u_svd = U @ np.diag(S) @ np.transpose(V)
            u_svd = utilities.convert_2d_to_3d(u_svd, dimX, dimY, dimT)
            u_svd = np.transpose(u_svd, (2,0,1))
            PSD_vals, k_vals = calculate_psd_1d(u_svd, dx=dx, dy=dy, DNS=True)
            PSD_vals_all[j] = PSD_vals

        PSD_avg = np.mean(PSD_vals_all, axis=0)

        psd_multi_recon[i] = PSD_avg
    
    #normalize k axis with integral length scale:
    k_vals = k_vals*integral_length_scale
    spectral_max=np.amax(psd_multi_recon)
    psd_multi = psd_multi_recon/spectral_max
    return psd_multi, k_vals



def calculate_PSD_r_vals_exp(u_fluc, rank_list, case, ensembles, plane):
    """
    Compute ensemble‑averaged 1‑D power‑spectral densities (PSD) for several
    SVD truncation ranks in an experimental Teetank velocity plane.

    The function:
        1. For each desired truncation rank in ``rank_list``  
           a. Loads reduced SVD factors (`U`, `S`, `V`) for each ensemble  
           b. Reconstructs the velocity field at rank ``rank``  
           c. Calculates a stream‑wise PSD via `calculate_psd_1d`  
           d. Averages PSDs across ensembles
        2. Stacks the averaged PSD curves into ``psd_multi`` and normalizes
           by their global maximum.
        3. Returns normalized PSDs and the wavenumber axis scaled by the
           integral length scale.

    Parameters
    ----------
    u_fluc : ndarray
        Full‑rank fluctuating velocity field; shape depends on experimental format
        (typically `(n_ensembles, ny, nx, nt)`).
    rank_list : list[int]
        List of SVD truncation ranks to evaluate (e.g. `[50, 100, 250]`).
    case : {"P25", "P50"}
        Experimental case identifier; controls spatial resolution and
        integral length scale.
    r : int
        Baseline rank used to load the pre‑computed SVD files.
    ensembles : list[int]
        experimental ensemble numbers to include in the PSD average.
    plane : int
        Velocity‑plane index (1 = H395, 2 = H390, …).

    Returns
    -------
    psd_multi : ndarray, shape (len(rank_list), len(k_vals))
        Normalized PSD curves for each rank in ``rank_list``.
    k_vals : ndarray
        Non‑dimensional wavenumber bins, scaled by the experiment’s
        integral length scale.

    Notes
    -----
    * Relies on helper functions in ``utilities`` and ``processdata`` to
      load reduced SVD data and compute PSDs.
    * Spatial resolutions are currently hard‑coded (`dx = dy = 1 mm`);
      adjust if experimental setup changes.
    * ``psd_multi`` is normalized to its global maximum for easier plotting.
    """
        

    #set experimental spatial resolution
    dx = 1e-3
    dy=1e-3
    if case=='P25':
        integral_length_scale=0.051
    else:
        integral_length_scale=0.068
    
    num_ens = len(ensembles)
    ensemble = ensembles[0]
    dimX, dimY, dimT = utilities.get_dims_exp_vel()
    u_fluc_nonan = np.nan_to_num(u_fluc[ensemble-1])
    u_fluc_nonan = np.transpose(u_fluc_nonan, (2,0,1))
    PSD_vals, k_vals = calculate_psd_1d(u_fluc_nonan, dx=dx, dy=dy)

    PSD_vals_all = np.zeros((num_ens, len(PSD_vals)))
    psd_multi_recon = np.zeros((len(rank_list), len(k_vals)))

    for i in range(len(rank_list)):
       
        rank = rank_list[i]
        
        PSD_vals_all = np.zeros((num_ens, len(k_vals)))
        for j in range(len(ensembles)):

            ensemble = ensembles[j]
            
            #get SVD matrices for velocity field
            U_tot_u_red, S_tot_u_red, U_tot_surf_red, S_tot_surf_red, V_tot_red = utilities.open_and_reduce_SVD(ensemble, case, rank, forecast=False, DNS=False, DNS_plane=None, DNS_surf=False, DNS_case='RE2500', exp=True, plane=plane)

            u_svd = U_tot_u_red @ np.diag(S_tot_u_red) @ np.transpose(V_tot_red[:,rank:2*rank])
            u_svd = utilities.convert_2d_to_3d(u_svd, dimY, dimX, dimT)
            u_fluc_nonan = np.nan_to_num(u_svd)
            u_svd = np.transpose(u_fluc_nonan, (2,0,1))
            #print(u_svd.shape)
            
            PSD_vals, k_vals = calculate_psd_1d(u_svd, dx=dx, dy=dy) 
            
            PSD_vals_all[j] = PSD_vals

        PSD_avg = np.mean(PSD_vals_all, axis=0)

        psd_multi_recon[i] = PSD_avg

    spectral_max=np.amax(psd_multi_recon)
    psd_multi = psd_multi_recon/spectral_max

    #normalize k axis with integral length scale:
    k_vals = k_vals*integral_length_scale

    return psd_multi, k_vals


def calculate_psd_1d(snapshots, dx=1.0, dy=1.0, DNS=False, time_avg=True):
    """
    Compute a 1-D streamwise power spectral density (PSD) from 2-D snapshots.

    Each snapshot is Hann-windowed along x, rFFT is taken along x, the
    magnitude-squared spectrum is averaged over y, and (optionally) averaged
    over time. For non-DNS data, zero-padding (+75 in x) increases spectral
    resolution.

    Parameters
    ----------
    snapshots : ndarray of shape (T, Nx, Ny)
        Sequence of 2-D fields (time, x, y).
    dx : float, optional
        Grid spacing in x (used for wavenumber scaling).
    dy : float, optional
        Grid spacing in y (kept for API symmetry; not used here).
    DNS : bool, optional
        If True, no zero-padding; else pad by 75 points in x.
    time_avg : bool, optional
        If True, average PSD over time; else return per-snapshot PSDs.

    Returns
    -------
    psd : ndarray, shape (K,) or (T, K)
        1-D PSD versus wavenumber (rad / unit length). If `time_avg` is True,
        shape is (K,); otherwise (T, K).
    k_mid : ndarray, shape (K,)
        Midpoint wavenumbers corresponding to `psd`.
    """
    num_snapshots, nx, ny = snapshots.shape
    if DNS:
        pad=0
    else:
        pad=75
    kx = np.fft.fftfreq(nx+pad, d=dx) * 2 * np.pi
    k_max = np.max(kx)
    k_bins = np.linspace(0, k_max, num=nx+pad//2)
    k_mid = 0.5 * (k_bins[:-1] + k_bins[1:])

    # Initialize an array for storing the radial PSD
    psd_all = np.zeros((num_snapshots, len(k_mid)))

    for i, snap in enumerate(snapshots):

        win = sig.windows.hann(nx,  sym=False)
        data_win = utilities.multiply_along_axis(snap, win, axis=0)
        
        data_fft = np.fft.rfft(data_win, n=len(k_mid)*2 -1, axis=0)
        PSD_vals = np.power(abs(data_fft),2)
        
        #avgeraging spectrum along y (axis 1)
        PSD_vals = np.mean(PSD_vals, axis=1)

        psd_all[i, :] = PSD_vals #psd_radial
        #psd_all.append(psd_radial)

    # Average PSD across all snapshots
    psd_all = np.array(psd_all)
    if time_avg:
        psd_mean = np.mean(psd_all, axis=0)
    else:
        psd_mean=psd_all
    # Compute the midpoints of wavenumber bins for output
    k_mid = 0.5 * (k_bins[:-1] + k_bins[1:])

    
    return psd_mean, k_mid


'''-----------------------------------------------------------------------------------------------------------------------------------'''

'''SHRED ANALYSIS FUNCTIONS'''

def SHRED_ensemble_DNS(r_vals, num_sensors, ens_start, ens_end, vel_planes, lags, full_planes=True, random_sampling=True, DNS_case='S2', criterion='MSE'):
    """
    Train and evaluate SHRED on multiple DNS ensembles and SVD ranks, then
    save reconstructed test snapshots to `.mat` files.

    Parameters
    ----------
    r_vals : list[int] or ndarray
        SVD truncation ranks to evaluate.
    num_sensors : int
        Number of randomly placed surface sensors.
    ens_start, ens_end : int
        Inclusive range of SHRED ensemble SEEDS.
    vel_planes : list[int]
        Indices of velocity planes to include (surface plane handled
        automatically).
    lags : int
        lag for the time series for input in LSTM. Standard number used is 52
    full_planes : bool, default True
        If True, stack all velocity planes; otherwise only those in
        ``vel_planes``.
    random_sampling : bool, default True
        If *True*, use random snapshot selection for train/val/test splits;
        if *False*, use a contiguous test block (forecast mode).
    DNS_case : {"S1", "S2"}, optional
        Case identifier; passed to utility loaders.
    criterion : {"MSE", ...}, default "MSE"
        Loss function name passed to :pyfunc:`models.fit`.

    Returns
    -------
    None
        This function is used for its side‑effects:
        * Trains SHRED models and shows loss plots.
        * Saves HDF5 files named
          ``SHRED_r{r}_{num_sensors}sensors_ens{p}.mat`` (or forecast
          variants) in `adr_loc`.

    Notes
    -----
    * Baseline full SVD rank ``r = 1000`` is hard‑coded.
    * Large raw DNS arrays must exist locally; not downloaded here.
    """

    DNS_case = utilities.case_name_converter(DNS_case)

    #extract surface and normalize
    surf = utilities.get_normalized_surface_DNS(DNS_case)
    
    #iterate SHRED ensembles p 
    for p in range(ens_start, ens_end+1):
        
        #iterate over ranks, if input is a list
        for q in range(len(r_vals)):
           
            r = r_vals[q]
            print("rank: ", r, "\n SHRED ensemble: ", p)
            
            #get SVD arrays and stack horizontally
            # includes all planes, plus SVD of surface on top
            U_tot, S_tot, V_tot = utilities.stack_svd_arrays_DNS(vel_planes, r, DNS_case=DNS_case)
            
            dimX, dimY, dimT = utilities.get_dims_DNS(DNS_case)

            #building array for input to SHRED
            #first insert V matrices fo all chosen velocity planes + surface 
            load_X = V_tot

            #assign random sensor placements
            sensor_locations_ne = np.random.choice(dimX*dimY, size=num_sensors, replace=False)
            print("sensor_loc: ", sensor_locations_ne)
            sensor_locations = np.arange(0,num_sensors,1, dtype=int)
    
            #stack sensor temporal data on top of the total V transposed array
            load_X = np.hstack((surf[sensor_locations_ne,:].T,load_X)) #horizontal stacking of arrays, columnwise, concatenation along 2nd axis
            n = (load_X).shape[0]   #number of snapshots in time series
            m = (load_X).shape[1]   #number of planes (plus surface) * number of SVD modes, plus number of sensors

            #creating a mask: grid with zeros expect the sensor points that's assigned with 1
            mask = np.zeros(dimX*dimY)
            for i in range(num_sensors):
                mask[sensor_locations_ne[i]]=1


            #choose mode for separation of data into training/validation/testing
            if random_sampling:
                n = (load_X).shape[0]

                train_indices = np.random.choice(n - lags, size=int(0.8*n), replace=False) #80% training
    
                mask = np.ones(n - lags) 
                mask[train_indices] = 0

                valid_test_indices = np.arange(0, n - lags)[np.where(mask!=0)[0]] #indices left for validation/testing
                valid_indices = valid_test_indices[::2] #pick every other index for validation (10%) 
                test_indices = valid_test_indices[1::2] #and the other (10%) for testing
                print("test indices: ", test_indices)
                n_test = test_indices.size

            else:
                #forecast, by choosing test data as one continuous chunk of the time series 
                #NOTE: note fully compatible with rest of code
                n = (load_X).shape[0]
                
                n_train_valid = int(n*0.90) #n - lags - n_test - n_valid
                
                n_test = n-lags - n_train_valid
                print("n_test: ", n_test)

                n_train = int((8/9)*n_train_valid)
                n_valid = n_train_valid - n_train

                #randomize training in the dedicated training-validation dataset
                train_indices = np.random.choice(n_train_valid, size=n_train, replace=False)
                #train_indices = np.arange(0, n_train)
                mask = np.ones(n_train_valid) 
                mask[train_indices] = 0
                valid_indices = np.arange(0, n_train_valid)[np.where(mask!=0)[0]]

                test_indices = np.arange(n_train_valid, n-lags)
            
           
            #scaling the input training data to SHRED
            sc = MinMaxScaler()
            sc = sc.fit(load_X[train_indices]) #computes min/max of training data for later scaling
            transformed_X = sc.transform(load_X) #use the previous scaling to fit and transform the training data


            ### Generate input sequences to a SHRED model
            all_data_in = np.zeros((n - lags, lags, num_sensors)) 
            for i in range(len(all_data_in)): #iterate and insert transformed traning data in sequences
                all_data_in[i] = transformed_X[i:i+lags, sensor_locations]

            
            ### Generate training validation and test datasets both for reconstruction of states and forecasting sensors
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            #print("device: ", device)

            train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
            valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
            test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)

           
            train_data_out = torch.tensor(transformed_X[train_indices + lags - 1], dtype=torch.float32).to(device)
            valid_data_out = torch.tensor(transformed_X[valid_indices + lags - 1], dtype=torch.float32).to(device)
            test_data_out = torch.tensor(transformed_X[test_indices + lags - 1], dtype=torch.float32).to(device)

            train_dataset = models.TimeSeriesDataset(train_data_in, train_data_out)
            valid_dataset = models.TimeSeriesDataset(valid_data_in, valid_data_out)
            test_dataset = models.TimeSeriesDataset(test_data_in, test_data_out)
            

            #DOING SHRED
            shred = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
            validation_errors = models.fit(shred, train_dataset, valid_dataset, criterion='MSE', batch_size=64, num_epochs=3000, lr=1e-3, verbose=True, patience=5)
            print("SHRED completed successfully!")
            #SHRED DONE


            #plot validation loss function
            if DNS_case=='RE2500':
                case_name = 'S2'
            else:
                case_name = 'S1'

            #plt.plot(training_errors, label='Train Loss')
            plt.plot(validation_errors, label='Validation Loss')
            plt.xlabel('Epoch', fontsize=16)         
            plt.ylabel('MSE', fontsize=16)
            plt.tick_params(axis='x', which='major', labelsize=13)
            plt.tick_params(axis='y', which='major', labelsize=13)
            plt.title("Loss per epoch, case " + case_name)
            plt.legend(fontsize=13)
            plt.grid(True)
            plt.show()

            #extract test reconstructions
            test_recons = sc.inverse_transform(shred(test_dataset.X).detach().cpu().numpy()) 
            test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy()) 
            print("test error: ", np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth))

            #save SHRED output
            SHRED_dict = {
                'test_recons': test_recons,
                'test_ground_truth': test_ground_truth,
                'test_indices' : test_indices,
            }
            
            
        
            
            if full_planes:
                plane_string ="_full_planes"

            else:
                plane_string = "_planes"
                for i in range(len(vel_planes)):
                    plane_string = plane_string + "_" +  str(vel_planes[i]) 

            if random_sampling:
                SHRED_fname = SHRED_DIR / ("SHRED_r"+ str(r) +"_" +str(num_sensors) +"sensors_ens" + str(p) + plane_string +".mat")
                if DNS_case=='RE1000':
                    SHRED_fname = SHRED_DIR / ("SHRED_RE1000_r"+ str(r) +"_" +str(num_sensors) +"sensors_ens" + str(p) + plane_string +".mat")
            else:
                if DNS_case=='RE1000':
                    SHRED_fname = SHRED_DIR / ("SHRED_FORECAST_RE1000_r"+ str(r) +"_" +str(num_sensors) +"sensors_ens" + str(p) + plane_string + ".mat")
                else:
                    SHRED_fname = SHRED_DIR / ("SHRED_FORECAST_r"+ str(r) +"_" +str(num_sensors) +"sensors_ens" + str(p) + plane_string + ".mat")
            with h5py.File(SHRED_fname, 'w') as f:
                for key, value in SHRED_dict.items():
                    f.create_dataset(key, data=value)



def SHRED_ensemble_exp(r_vals, num_sensors, X, ens_start, ens_end, case, experiment_ens, lags=52, exp_plane='H390', random_sampling=True, criterion='MSE'):
    """
    Train SHRED on experimental data from the turbulent 'T-tank' for a single velocity plane
    plus the surface, across multiple SVD ranks and experimental ensemble seeds, then
    save reconstructed test snapshots to `.mat` files.

    
    Parameters
    ----------
    r_vals : list[int] or ndarray
        SVD truncation ranks to evaluate.
    num_sensors : int
        Number of randomly placed surface sensors.
    X : ndarray
        Surface‑elevation time series (shape `(n_points, nt)`).
    ens_start, ens_end : int
        Inclusive range of SHRED ensemble seeds.
    case : {"E1", "E2"}
        experimental case identifier.
    experiment_ens : int
        Experimental ensemble number .
    lags : int, default 52
        lag for the time series for input in LSTM. Standard number used is 52
    exp_plane : {"H390", "H375", "H350", "H300"}, optional
        Velocity‑plane label to reconstruct. Standard is 'H390' (upper layer, 1 cm below surface)
    random_sampling : bool, default True
        If *True*, use random snapshot selection for train/val/test splits;
        if *False*, use a contiguous test block (forecast mode).
    criterion : {"MSE", ...}, default "MSE"
        Loss function used in `models.fit`.

    Returns
    -------
    None
        Generates validation‑loss plots and writes HDF5 files named
        ``Teetank_SHRED_*`` (or forecast variants) to `adr_loc`.

    Notes
    -----
    * Only **one** velocity plane is processed at a time; `full_planes`
      logic is DNS‑only.
    * `X` must contain the full‐length sensor trajectories covering all
      time steps of the reduced SVD matrices.
    """
    #ONLY RUN SHRED FOR 1 PLANE PLUS SURFACE AT THE SAME TIME!
    
    case = utilities.case_name_converter(case)

    
    #exctract full SVD matrices for this plane (velocity plane stacked with surface)
    U_tot_u, S_tot_u, U_tot_surf, S_tot_surf, V_tot = utilities.open_SVD(experiment_ens, vel_fluc=False, variable='u', exp=True, experimental_case=case,  DNS_new=False, DNS_plane=None, DNS_surf=False, experimental_plane=exp_plane)
    
    #extract surface grid
    X_surf, Y_surf, X_vel, Y_vel = utilities.get_mesh_exp(case, exp_plane)
    
    #iterate SHRED ensembles p 
    for p in range(ens_start, ens_end+1):
        
        #iterate over ranks, if input is a list
        for q in range(len(r_vals)):
            r = r_vals[q]
            print("rank: ", r, "\n SHRED ensemble: ", p)
            
            #reduce SVD matrices (only need V matrix)
            U_tot_u_red, S_tot_u_red, V_tot_red = utilities.reduce_SVD(U_tot_u, S_tot_u, V_tot, levels=2, rank=r, Tee=True, surf=False)
            
            dimX = X_surf.shape[0] 
            dimY = X_surf.shape[1]

            #building array for input to SHRED
            #first insert V matrix for chosen plane + corresponding surface 
            load_X = V_tot_red
            
            #assign random sensor placements
            sensor_locations_ne = np.random.choice(dimX*dimY, size=num_sensors, replace=False)
            print("sensor_loc: ", sensor_locations_ne)
            sensor_locations = np.arange(0,num_sensors,1, dtype=int)
    
            #stack sensor temporal data on top of the total V transposed array
            load_X = np.hstack((X[sensor_locations_ne,:].T,load_X)) #horizontal stacking of arrays, columnwise, concatenation along 2nd axis
            n = (load_X).shape[0] #number of snapshots in time series
            m = (load_X).shape[1] #number of planes (surface + one velocity plane) * number of SVD modes, plus number of sensors

            mask = np.zeros(dimX*dimY)
            for i in range(num_sensors):
                mask[sensor_locations_ne[i]]=1

            #choose mode for separation of data into training/validation/testing
            if random_sampling:
                n = (load_X).shape[0]
                train_indices = np.random.choice(n - lags, size=int(0.8*n), replace=False) #80% training
                
                mask = np.ones(n - lags) 
                mask[train_indices] = 0

                valid_test_indices = np.arange(0, n - lags)[np.where(mask!=0)[0]] #indices left for validation/testing
                valid_indices = valid_test_indices[::2] #pick every other index for validation (10%)
                test_indices = valid_test_indices[1::2] #and the other (10%) for testing
                print("test indices: ", test_indices)
                n_test = test_indices.size

                
            else:
                #forecast, by choosing test data as one continuous chunk of the time series
                #this forecast is only within one experimental ensemble case, 
                # not e.g. training on some cases and testing on others
                # #NOTE: note fully compatible with rest of code 
                n = (load_X).shape[0]
                
                n_valid = int(n*0.1)
                n_train = int(n*0.8) #n - lags - n_test - n_valid
                print("n_train: ", n_train)
                n_test = n-lags - n_valid - n_train
                print("n_test: ", n_test)
                train_indices = np.arange(0, n_train)
                valid_indices = np.arange(n_train, n_train + n_valid)
                test_indices = np.arange(n_train + n_valid, n - lags)

    
            #scaling the input training data to SHRED
            sc = MinMaxScaler()
            sc = sc.fit(load_X[train_indices]) #computes min/max of training data for later scaling
            transformed_X = sc.transform(load_X) #use the previous scaling to fit and transform the training data


            ### Generate input sequences to a SHRED model
            all_data_in = np.zeros((n - lags, lags, num_sensors)) 
            for i in range(len(all_data_in)): #iterate and insert transformed traning data in sequences
                all_data_in[i] = transformed_X[i:i+lags, sensor_locations]


            ### Generate training validation and test datasets both for reconstruction of states and forecasting sensors
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            #print("device: ", device)

            train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
            valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
            test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)

            ### -1 to have output be at the same time as final sensor measurements
            train_data_out = torch.tensor(transformed_X[train_indices + lags - 1], dtype=torch.float32).to(device)
            valid_data_out = torch.tensor(transformed_X[valid_indices + lags - 1], dtype=torch.float32).to(device)
            test_data_out = torch.tensor(transformed_X[test_indices + lags - 1], dtype=torch.float32).to(device)

            train_dataset = models.TimeSeriesDataset(train_data_in, train_data_out)
            valid_dataset = models.TimeSeriesDataset(valid_data_in, valid_data_out)
            test_dataset = models.TimeSeriesDataset(test_data_in, test_data_out)
            
    

            #DOING SHRED

            shred = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
            validation_errors = models.fit(shred, train_dataset, valid_dataset, batch_size=64, num_epochs=3000, lr=1e-3, verbose=True, patience=5)
            print("SHRED completed successfully!")
            #SHRED DONE
            

            #plot validation loss function
            if case=='P50':
                case_name ='E2'
            else:
                case_name = 'E1'
            #plt.plot(training_errors, label='Train Loss')
            plt.plot(validation_errors[2:], label='Validation Loss')
            plt.xlabel('Epoch', fontsize=16)         
            plt.ylabel('MSE', fontsize=16)
            plt.tick_params(axis='x', which='major', labelsize=13)
            plt.tick_params(axis='y', which='major', labelsize=13)
            plt.title("Loss per epoch, case " + case_name)
            plt.legend(fontsize=13)
            plt.grid(True)
            plt.show()
        
           #extract test reconstructions
            test_recons = sc.inverse_transform(shred(test_dataset.X).detach().cpu().numpy()) 
            test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy()) 
            print(np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth))

            #save SHRED output
            SHRED_dict = {
                'test_recons': test_recons,
                'test_ground_truth': test_ground_truth,
                'test_indices' : test_indices,
            }
            
            if random_sampling:
                SHRED_fname = SHRED_DIR / ("Teetank_SHRED_new_ens"+ str(experiment_ens) + "_"+ case + "_" + exp_plane + "_r"+ str(r) +"_" +str(num_sensors) +"sensors_ens" + str(p) +".mat")
            else:
                SHRED_fname = SHRED_DIR / ("SHRED_r"+ str(r) +"_" + case + "_" +exp_plane +"_" +str(num_sensors) +"sensors_ens" + str(p) +"_prediction.mat")
            with h5py.File(SHRED_fname, 'w') as f:
                for key, value in SHRED_dict.items():
                    f.create_dataset(key, data=value)


'''------------------------------------------------------------------------------------------------------------------------------'''

'''POST-SHRED ANALYSIS FUNCTIONS AND ERROR METRIC ANALYSIS'''


def calculate_instantaneous_rms_profile(DNS_case, SHRED_ens, rank, num_sensors):
    '''compute depth-dependent RMS velocity profile without time averaging'''
    
    DNS_case = utilities.case_name_converter(DNS_case)
    #open and stack SVD planes
    if DNS_case=='RE2500':
        tot_num_planes=76
    else:
        tot_num_planes=57
    
    #open and reduce SVD


    stack_planes = np.arange(1, tot_num_planes+1)

    U_tot_red, S_tot_red, V_tot_red = utilities.stack_svd_arrays_DNS(stack_planes, rank, DNS_case=DNS_case)     

    #open SHRED ensemble
    V_tot_recons, V_tot_svd, test_indices = utilities.open_SHRED(exp_ens=None, case=DNS_case, rank=rank, num_sensors=num_sensors, SHRED_ens=SHRED_ens, plane_list=None, DNS=True, 
                                                                     full_planes=True, forecast=False)
    num_test_snaps = len(test_indices)

    #construct velocity fields from SVD
    rms_gt = np.zeros((tot_num_planes, num_test_snaps))
    rms_recons = np.zeros((tot_num_planes, num_test_snaps))

    for j in range(tot_num_planes):
        plane = j+1
        plane_index = plane #shift with 1 to compensate for surface elevation when loading V matrices
       #print("plane: ", plane)
        
        #xtract test snapshots and reconstructions
        u_fluc=None
        u_fluc_test, u_svd_test, u_recons_test, u_fluc_full = utilities.get_test_imgs_SHRED_DNS(DNS_case, plane, plane_index, u_fluc, V_tot_recons, test_indices,  rank, 
                                                                                                    num_sensors,U_tot_red, S_tot_red, V_tot_red, open_svd=False, lags=52, 
                                                                                                    forecast=False, surface=False, no_input_u_fluc=True)
        u_gt = utilities.convert_3d_to_2d(u_fluc_test) #ground truth velocity field
        u_recons = utilities.convert_3d_to_2d(u_recons_test)
        rms_gt[j] = utilities.RMS_plane(u_gt)
        rms_recons[j] = utilities.RMS_plane(u_recons)

    #save ground truth and reconstruction RMS profiles for all test snapshots, in a dictionary
    rms_dict = {
        'rms_gt' : rms_gt,
        'rms_recons' : rms_recons
    }
    
    if DNS_case=='RE2500':
        rms_fname = METRICS_DIR / ("instantaneous_rms_SHRED"+ str(SHRED_ens) + "_RE2500.mat")
    else:
        rms_fname = METRICS_DIR / ("instantaneous_rms_SHRED"+ str(SHRED_ens) + "_RE1000.mat")

    with h5py.File(rms_fname, 'w') as f:
        for key, value in rms_dict.items():
            f.create_dataset(key, data=value)
    
    return rms_gt, rms_recons


def calculate_instantaneous_rms_profile_exp(case, experimental_ens, SHRED_ens, rank, num_sensors):
    '''compute depth-dependent RMS velocity profile without time averaging'''
    
    case = utilities.case_name_converter(case)
    
    #open and stack SVD planes

    tot_num_planes=4
    vel_planes = [1,2,3,4,5]
    V_tot_recons, V_tot_svd, test_indices = utilities.open_SHRED(experimental_ens, case, rank, num_sensors, SHRED_ens, vel_planes, DNS=False,  exp_plane='H390', full_planes=True, forecast=False)
    num_test_snaps = len(test_indices)

    #construct velocity fields from SVD
    rms_gt = np.zeros((tot_num_planes, num_test_snaps))
    rms_recons = np.zeros((tot_num_planes, num_test_snaps))

    for j in range(tot_num_planes):
        plane_index = j+2 #first velocity plane is ignored, file doesn't exist

        print("plane: ", plane_index)
        planes = ['H395', 'H390', 'H375', 'H350', 'H300']
        plane = planes[plane_index-1]
        print("plane: ", plane)
        
        X_surf, Y_surf, X_vel, Y_vel = utilities.get_mesh_exp(case, plane)


        #open SHRED for this plane-surface-pairing, T-TANK-ensemble and SHRED ensemble
        V_tot_recons, V_tot_svd, test_indices = utilities.open_SHRED(experimental_ens, case, rank, num_sensors, SHRED_ens, vel_planes, DNS=False,  exp_plane=plane, full_planes=True, forecast=False)
        num_test_snaps = len(test_indices)
        #get SVDs correctly
        U_tot_u_red, S_tot_u_red, U_tot_surf_red, S_tot_surf_red, V_tot_red = utilities.open_and_reduce_SVD(experimental_ens, case, rank, forecast=False, DNS=False, DNS_plane=None,
                                                                                                                   DNS_surf=False, exp=True, plane=plane)

        #extract test snapshots and reconstructions
        surf_fluc=None
        u_fluc=None
        u_fluc_test, u_svd_test, u_recons_test, u_fluc_full = utilities.get_test_imgs_SHRED_exp(plane, surf_fluc, u_fluc, V_tot_recons, V_tot_svd, test_indices, X_surf, X_vel, experimental_ens, case, rank, 
                                                                                                            SHRED_ens, num_sensors, U_tot_u_red, S_tot_u_red, V_tot_red = V_tot_red, open_svd=False, lags=52, forecast=False, 
                                                                                                            surface=False,no_input_u_fluc=True)

        #calculate error metrics for this plane and ensemble case

        u_gt = utilities.convert_3d_to_2d(u_fluc_test)
        u_recons = utilities.convert_3d_to_2d(u_recons_test)
        rms_gt[j] = utilities.RMS_plane(u_gt)
        rms_recons[j] = utilities.RMS_plane(u_recons)


        u_gt = utilities.convert_3d_to_2d(u_fluc_test)
        u_recons = utilities.convert_3d_to_2d(u_recons_test)
        rms_gt[j] = utilities.RMS_plane(u_gt)
        rms_recons[j] = utilities.RMS_plane(u_recons)

    rms_dict = {
        'rms_gt' : rms_gt,
        'rms_recons' : rms_recons
    }
    
    rms_fname = METRICS_DIR / ("instantaneous_rms_SHRED_"+case + "_SHREDens_" +  str(SHRED_ens) + "_teeEns_" + str(experimental_ens) + ".mat")

    with h5py.File(rms_fname, 'w') as f:
        for key, value in rms_dict.items():
            f.create_dataset(key, data=value)
    
    return rms_gt, rms_recons


def open_instantaneous_rms_profile(DNS_case, SHRED_ens):
    '''opens the instantaneous (non-time-averaged) rms profile for a specific DNS case and SHRED ensemble'''
    if DNS_case=='RE2500':
        rms_fname = METRICS_DIR / ("instantaneous_rms_SHRED"+ str(SHRED_ens) + "_RE2500.mat")
    else:
        rms_fname = METRICS_DIR / ("instantaneous_rms_SHRED"+ str(SHRED_ens) + "_RE1000.mat")
    with h5py.File(rms_fname, 'r') as rms_dict:
        rms_gt = np.array(rms_dict['rms_gt'])
        rms_recons = np.array(rms_dict['rms_recons'])

    return rms_gt, rms_recons


def open_instantaneous_rms_profile_exp(exp_case, experimental_ens, SHRED_ens):
    '''opens the instantaneous (non-time-averaged) rms profile for a specific DNS case and SHRED ensemble'''
    
    rms_fname = METRICS_DIR / ("instantaneous_rms_SHRED_"+exp_case + "_SHREDens_" +  str(SHRED_ens) + "_teeEns_" + str(experimental_ens) + ".mat")
    with h5py.File(rms_fname, 'r') as rms_dict:
        rms_gt = np.array(rms_dict['rms_gt'])
        rms_recons = np.array(rms_dict['rms_recons'])

    return rms_gt, rms_recons


def normalized_mean_square_error(gt, recon):
    """Compute normalized mean square error (NMSE) between ground truth and reconstruction."""
    mse = np.mean((gt - recon) ** 2, axis=(1, 2, 3))  # Mean over spatial and time dimensions
    norm_factor = np.mean(gt ** 2, axis=(1, 2, 3))  # Normalization factor
    return mse / norm_factor



def power_spectral_density_error(gt, recon, num_scales, DNS=True, DNS_case='RE2500'):
    """Compute normalized mean error in the power spectral density up to a defined cutoff wavenumber."""

    nx, ny, nt = recon.shape
    bins = nx // 2
    if DNS:
        dimX, dimY, dimT = utilities.get_dims_DNS(DNS_case)
        nt2 = dimT
        dx=2*np.pi/dimX
        dy=2*np.pi/dimY
        if DNS_case=='RE2500':
            cutoff_index = 13
        else:
            cutoff_index = 11 
    else:
        nt2 = 900
        dx=1e-3
        dy=dx
        cutoff_index = 7

            
    gt = np.transpose(gt, (2,0,1))
    gt_fft, k_vals = calculate_psd_1d(gt, dx, dy, DNS)
    recon = np.transpose(recon, (2,0,1))
    recon_fft, k_vals = calculate_psd_1d(recon, dx, dy, DNS)


    # Bin indices
    k_bins = np.linspace(0, k_vals.max(), bins)
    psd_diff = []

    gt_int = scipy.integrate.simpson(gt_fft[:cutoff_index], x=k_vals[:cutoff_index])
    
    recon_int = scipy.integrate.simpson(recon_fft[:cutoff_index], x=k_vals[:cutoff_index])
    psd_error = np.abs(gt_int - recon_int)/gt_int
    # Normalize error by ground truth PSD
    #psd_error = np.sum(psd_diff) / np.sum(gt_fft)

    return psd_error



def power_spectral_density_compare(gt, recon, num_scales, DNS=True, DNS_case='RE2500'):
    nx, ny, nt = recon.shape
    bins = nx // 2
    if DNS:
        dimX, dimY, dimT = utilities.get_dims_DNS(DNS_case)
        nt2 = dimT
        dx=2*np.pi/dimX
        dy=2*np.pi/dimY
        if DNS_case=='RE2500':
            cutoff_index = 13
        else:
            cutoff_index = 11 
    else:
        nt2 = 900
        dx=1e-3
        dy=dx
        cutoff_index = 7
            

    gt = np.transpose(gt, (2,0,1))
    gt_fft, k_vals = calculate_psd_1d(gt, dx, dy, DNS)
    recon = np.transpose(recon, (2,0,1))
    recon_fft, k_vals = calculate_psd_1d(recon, dx, dy, DNS)

    # Bin indices
    k_bins = np.linspace(0, k_vals.max(), bins)
    psd_diff = []

    return gt_fft, recon_fft, k_vals



def calculate_psd_rank_dependence(r_vals, case, DNS, vel_planes, plane_index, num_sensors, SHRED_ens, experimental_ens):
    """
    Compute rank-dependent, normalized 1D PSDs for a selected plane and compare
    ground truth vs SVD-compressed vs SHRED-reconstructed fields.

    For each rank r in `r_vals`, this loads/constructs the SVD truncation and the
    corresponding SHRED reconstruction (DNS or experimental), computes the PSD
    versus wavenumber, normalizes each spectrum by the ground-truth spectral
    maximum, and rescales k by the integral length scale.

    Parameters
    ----------
    r_vals : Sequence[int]
        SVD ranks to evaluate.
    case : str
        Case identifier (e.g., 'RE1000', 'RE2500', 'P25', 'P50').
    DNS : bool
        True for DNS data; False for experimental data.
    vel_planes : Sequence[int]
        Plane indices available; the plane at `plane_index` is analyzed.
    plane_index : int
        Index selecting the plane within `vel_planes` (DNS) or plane list (exp).
    num_sensors : int
        Number of surface sensors used by SHRED.
    SHRED_ens : int
        SHRED ensemble identifier.
    experimental_ens : int
        Experimental ensemble identifier (ignored if DNS=True).

    Returns
    -------
    psd_gt : np.ndarray, shape (K,)
        Ground-truth normalized PSD for the selected plane.
    psd_svd_r : np.ndarray, shape (len(r_vals), K)
        Normalized PSDs of SVD-compressed fields for each rank.
    psd_recons_r : np.ndarray, shape (len(r_vals), K)
        Normalized PSDs of SHRED reconstructions for each rank.
    k_vals : np.ndarray, shape (K,)
        Dimensionless wavenumber vector (k * L_infty).
    """      

    for i in range(len(r_vals)):
        r = r_vals[i]
        print("rank: ", r)
        if DNS:
            stack_planes=vel_planes
            plane = vel_planes[plane_index]
            integral_length_scale=utilities.get_integral_length_scale(case)
            U_tot_red, S_tot_red, V_tot_red = utilities.stack_svd_arrays_DNS(stack_planes, r,  case)                                                  
  
            
            V_tot_recons, V_tot_svd, test_indices = utilities.open_SHRED(None, case, r, num_sensors, SHRED_ens, stack_planes, DNS=True, 
                                                                     full_planes=False, forecast=False)
            u_fluc_test, u_svd_test, u_recons_test, u_fluc_full = utilities.get_test_imgs_SHRED_DNS(case, plane, plane_index+1, None, V_tot_recons, test_indices, r, 
                                                                                                    num_sensors,U_tot_red, S_tot_red, V_tot_red, open_svd=False, lags=52, 
                                                                                                    forecast=False, surface=False, no_input_u_fluc=True)
            gt_fft, recon_fft, k_vals = power_spectral_density_compare(u_fluc_test, u_recons_test, 3, DNS=True, DNS_case=case)
            gt_fft, svd_fft, k_vals = power_spectral_density_compare(u_fluc_test, u_svd_test, 3, DNS=True, DNS_case=case)
            spectral_max=np.amax(gt_fft)
            gt_psd = gt_fft/spectral_max
            recon_psd = recon_fft/spectral_max
            svd_psd = svd_fft/spectral_max
            k_vals = integral_length_scale*k_vals

        
        else:

            planes = ['H395', 'H390', 'H375', 'H350', 'H300']
            plane = planes[plane_index]
            X_surf, Y_surf, X_vel, Y_vel = utilities.get_mesh_exp(case, plane)
            #open SHRED for this plane-surface-pairing, Tee-ensemble and SHRED ensemble
            V_tot_recons, V_tot_svd, test_indices = utilities.open_SHRED(experimental_ens, case, r, num_sensors, SHRED_ens, vel_planes, DNS=False,  exp_plane=plane, full_planes=True, forecast=False)
    
            #get SVDs correctly
            U_tot_u_red, S_tot_u_red, U_tot_surf_red, S_tot_surf_red, V_tot_red = utilities.open_and_reduce_SVD(experimental_ens, case, r, forecast=False, DNS=False, DNS_plane=None,
                                                                                                                   DNS_surf=False, exp=True, plane=plane)
            surf_fluc=None
            u_fluc_test, u_svd_test, u_recons_test, u_fluc_full = utilities.get_test_imgs_SHRED_exp(plane, surf_fluc, None, V_tot_recons, V_tot_svd, test_indices, X_surf, X_vel, experimental_ens, case, r, 
                                                                                                            SHRED_ens, num_sensors, U_tot_u_red, S_tot_u_red, V_tot_red, open_svd=False, lags=52, forecast=False, 
                                                                                                            surface=False,no_input_u_fluc=True)
            gt_fft, recon_fft, k_vals = power_spectral_density_compare(u_fluc_test, u_recons_test, 3, DNS=False)
            gt_fft, svd_fft, k_vals = power_spectral_density_compare(u_fluc_test, u_svd_test, 3, DNS=False)
            spectral_max=np.amax(gt_fft)
            gt_psd = gt_fft/spectral_max
            recon_psd = recon_fft/spectral_max
            svd_psd = svd_fft/spectral_max
            integral_length_scale=0.051
            k_vals = integral_length_scale*k_vals

        if i==0:
            psd_recons_r = np.zeros((len(r_vals),len(k_vals)))
            psd_svd_r = np.zeros((len(r_vals),len(k_vals)))
            psd_gt =  gt_psd
  
        psd_recons_r[i] = recon_psd
        psd_svd_r[i] = svd_psd
    
    return psd_gt, psd_svd_r, psd_recons_r, k_vals
        



def power_spectral_density_error_time_series(gt, recon, num_scales, test_indices, DNS=True, DNS_case='RE2500'):
    """Compute normalized mean error in the power spectral density, for each test snapshot given by test_indices"""

    nx, ny, nt = recon.shape
    bins = nx // 2
    if DNS:
        dimX, dimY, dimT = utilities.get_dims_DNS(DNS_case)
        nt2 = dimT
        dx=2*np.pi/dimX
        dy=2*np.pi/dimY
        if DNS_case=='RE2500':
            cutoff_index = 13
        else:
            cutoff_index = 11 
    else:
        nt2 = 900
        dx=1e-3
        dy=dx
        cutoff_index = 7

           

    gt = np.transpose(gt, (2,0,1))
    gt=gt[test_indices]
    gt_fft, k_vals = calculate_psd_1d(gt, dx, dy, DNS, time_avg=False)
    recon = np.transpose(recon, (2,0,1))
    recon_fft, k_vals = calculate_psd_1d(recon, dx, dy, DNS, time_avg=False)
    num_snaps=gt_fft.shape[0]
    
    psd_error = np.zeros(num_snaps)
    for i in range(num_snaps):
        gt_int = scipy.integrate.simpson(gt_fft[i,:cutoff_index], x=k_vals[:cutoff_index])
    
        recon_int = scipy.integrate.simpson(recon_fft[i,:cutoff_index], x=k_vals[:cutoff_index])
        psd_error[i] = np.abs(gt_int - recon_int)/gt_int

    return psd_error



def mutual_information(hgram):
    """ Mutual information for joint histogram
    """
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


def calculate_error_metrics(DNS_case, rank, vel_planes, num_sensors, SHRED_ensembles, lags=52, forecast=False, full_planes=True):
    """
    Compute depth-dependent reconstruction metrics for SHRED runs on DNS
    data and save them to `.mat` files.

    Parameters
    ----------
    DNS_case : {"S1", "S2"}
        DNS dataset identifier.
    rank : int
        SVD truncation rank used for both SVD baseline and SHRED model.
    vel_planes : list[int]
        Velocity-plane indices for which to compute metrics.
    num_sensors : int
        Number of surface sensors present in the SHRED run (used for
        filename bookkeeping).
    SHRED_ensembles : list[int]
        SHRED ensemble seeds to process.
    lags : int, default 52
        LSTM lag (needed for test-index
        alignment).
    forecast : bool, default False
        If *True*, load forecast-mode SHRED outputs; otherwise standard
        reconstruction mode.
    full_planes : bool, default True
        If *True*, SVD stacks **all** planes; if *False*, only
        ``vel_planes``.  Controls plane indexing logic.

    Returns
    -------
    None
        Writes an HDF5 file per ensemble

        * ``RMS_recons``  
        * ``RMS_true``  
        * ``MSE_z``  
        * ``ssim``  
        * ``psnr``  
        * ``psd``  

    """

    DNS_case = utilities.case_name_converter(DNS_case)
    #useful to define total number of planes per case
    #helps picking out correct planes if we only want to
    #plot some (but not all) planes
    
    if DNS_case=='RE2500':
        tot_num_planes=76
    else:
        tot_num_planes=57
    
    #open and reduce SVD
  
    if full_planes:
        stack_planes = np.arange(1, tot_num_planes+1)
    else:
        stack_planes=vel_planes 

    U_tot_red, S_tot_red, V_tot_red = utilities.stack_svd_arrays_DNS(stack_planes, rank, DNS_case=DNS_case)                                                  

    #then iterate SHRED ensembles

    print("start SHRED ensemble looping")
    for i in range(len(SHRED_ensembles)):
        ensemble = SHRED_ensembles[i]
        print("ensemble: ", ensemble)

    
        V_tot_recons, V_tot_svd, test_indices = utilities.open_SHRED(None, DNS_case, rank, num_sensors, ensemble, vel_planes, DNS=True, 
                                                                     full_planes=full_planes, forecast=forecast)
        
        num_test_snaps = len(test_indices)
        RMS_recons = np.zeros(len(vel_planes))
        RMS_true = np.zeros((len(vel_planes)))

        #initialize vertical profile error metric arrays
        MSE_z = np.zeros(len(vel_planes))
        ssim_values = np.zeros(len(vel_planes))
        psnr_values = np.zeros(len(vel_planes))
        psd_error = np.zeros(len(vel_planes))

        #calculate plane-by-plane
        for j in range(len(vel_planes)):
            plane = vel_planes[j]
            if full_planes and len(vel_planes) < tot_num_planes:
                plane_index = vel_planes[j]
            else:
                plane_index = j+1 #shift with 1 to compensate for surface elevation when loading V matrices
            
            print("plane: ", plane)
            
            u_fluc=None

            #extract test snapshots for this plane
            u_fluc_test, u_svd_test, u_recons_test, u_fluc_full = utilities.get_test_imgs_SHRED_DNS(DNS_case, plane, plane_index, u_fluc, V_tot_recons, test_indices, rank, 
                                                                                                    num_sensors,U_tot_red, S_tot_red, V_tot_red, open_svd=False, lags=lags, 
                                                                                                    forecast=False, surface=False, no_input_u_fluc=True)

            #calculate error metrics for this plane and SHRED ensemble case

            #RMS 
            u_recons = utilities.convert_3d_to_2d(u_recons_test)
            u_truth = utilities.convert_3d_to_2d(u_fluc_test)

            RMS_recons[j] = utilities.get_RMS(u_recons)
            RMS_true[j] =  utilities.get_RMS(u_truth)


            #Normalized Mean squared error (NMSE)
            print("calc MSE")
            mse = np.mean((u_truth - u_recons) ** 2, axis=(0,1))  # Mean over spatial and time dimensions
            norm_factor = np.mean(u_truth ** 2, axis=(0, 1))
            MSE_z[j] = mse / norm_factor


            #SSIM
            print("calc SSIM")
            ssim_snapshots = [
                ssim(u_fluc_test[:, :, t], u_recons_test[:, :, t], data_range=u_fluc_test[ :, :, t].max() - u_fluc_test[ :, :, t].min())
            for t in range(num_test_snaps)
            ]
            ssim_values[j] = np.mean(ssim_snapshots)
            print("ssim_vals: ", ssim_values[j])

            #PSNR
            print("calc PSNR")
            psnr_snapshots = [
            psnr(
                u_fluc_test[:, :, t],
                u_recons_test[:, :, t],
                data_range=u_fluc_test[ :, :, t].max() - u_fluc_test[:, :, t].min()
            )
            for t in range(num_test_snaps)
            ]
            psnr_values[j] = np.mean(psnr_snapshots)  # Average over time
            print("PSNR val: ", psnr_values[j])

            #PSD
            print("calc PSD")
            psd_error[j] = power_spectral_density_error(u_fluc_full, u_recons_test, 3, DNS=True, DNS_case=DNS_case)
            print("PSD error: ", psd_error[j])
        
        #save vertical error metric profiles in dictionary
        err_dict = {
            'RMS_recons' : RMS_recons,
            'RMS_true' : RMS_true,
            'MSE_z' : MSE_z,
            'ssim' : ssim_values,
            'psnr' : psnr_values,
            'psd' : psd_error
        }


        
        if full_planes:
            plane_string ="_full_planes"
        else:  
            plane_string = "_planes"
            for i in range(len(vel_planes)):
                plane_string = plane_string + "_" +  str(vel_planes[i])
        if forecast:
            fcast ="_forecast_"
        else:
            fcast="_"
       
        if DNS_case=='RE2500': 
            err_fname = METRICS_DIR / ("err_metrics" + fcast +"r"+str(rank)+"_sens"+str(num_sensors)+ "_ens"+ str(ensemble)+ plane_string +  ".mat")
        else:
            err_fname = METRICS_DIR / ("err_metrics_RE1000" + fcast +"r"+str(rank)+"_sens"+str(num_sensors)+ "_ens"+ str(ensemble)+ plane_string +  ".mat")
  
        with h5py.File(err_fname, 'w') as f:
            for key, value in err_dict.items():
                f.create_dataset(key, data=value)
        print("saved successfully!")



def calculate_error_metrics_exp(case, rank, vel_planes, num_sensors, SHRED_ensembles, experimental_ensembles, lags=52, forecast=False, full_planes=True):
    """
    Compute vertical profiles of reconstruction-error metrics for SHRED
    runs on Teetank experimental data and save the results to `.mat` files.

    Parameters
    ----------
    case : {"E1", "E2"}
        Teetank case identifier 
    rank : int
        SVD truncation rank used for the comparison.
    vel_planes : list[int]
        List of velocity planes (1-based: 1 = H395, …) to evaluate.
    num_sensors : int
        Number of surface sensors used in the SHRED model.
    SHRED_ensembles : list[int]
        SHRED ensemble seeds to process.
    experimental_ensembles : list[int]
        Teetank ensemble indices providing the underlying SVD data.
    lags : int, default 52
        Length of LSTM input sequences (needed for test-index alignment).
    forecast : bool, default False
        If *True*, load SHRED forecast outputs; otherwise reconstruction.
    full_planes : bool, default True
        If *True*, metrics are computed for all available planes; if
        *False*, only those in ``vel_planes`` (affects file naming).

    Returns
    -------
    None
        Writes one HDF5 (`.mat`) file per SHRED ensemble containing the
        following datasets:

        * ``RMS_recons``  
        * ``RMS_true``  
        * ``MSE_z``  
        * ``ssim``  
        * ``psnr``  
        * ``psd``  

    """

    case = utilities.case_name_converter(case)

    print("start ensemble looping")

    for k in range(len(experimental_ensembles)):
        
        experimental_ens = experimental_ensembles[k]
        #open and reduce SVD matrices for the given experimental ensemble

        for i in range(len(SHRED_ensembles)):
            ensemble = SHRED_ensembles[i]
            print("ensemble: ", ensemble)

            
            RMS_recons = np.zeros(len(vel_planes))
            RMS_true = np.zeros((len(vel_planes)))

            MSE_z = np.zeros(len(vel_planes))
            ssim_values = np.zeros(len(vel_planes))
            psnr_values = np.zeros(len(vel_planes))
            psd_error = np.zeros(len(vel_planes))

            for j in range(len(vel_planes)):
                plane = vel_planes[j]
                print("plane: ", plane)
                planes = ['H395', 'H390', 'H375', 'H350', 'H300']
                plane = planes[vel_planes[j]-1]
                print("Plane: ", plane)
                X_surf, Y_surf, X_vel, Y_vel = utilities.get_mesh_exp(case, plane)
                #open SHRED for this plane-surface-pairing, T-tank-ensemble cases and SHRED ensemble
                V_tot_recons, V_tot_svd, test_indices = utilities.open_SHRED(experimental_ens, case, rank, num_sensors, ensemble, vel_planes, DNS=False,  exp_plane=plane, full_planes=full_planes, forecast=forecast)
                num_test_snaps = len(test_indices)
                
                #get SVDs correctly
                U_tot_u_red, S_tot_u_red, U_tot_surf_red, S_tot_surf_red, V_tot_red = utilities.open_and_reduce_SVD(experimental_ens, case, rank, forecast=forecast, DNS=False, DNS_plane=None,
                                                                                                                   DNS_surf=False, exp=True, plane=plane)

                #Extract test snapshots from SHRED run
            
                u_fluc_test, u_svd_test, u_recons_test, u_fluc_full = utilities.get_test_imgs_SHRED_exp(plane, None, None, V_tot_recons, V_tot_svd, test_indices, X_surf, X_vel, experimental_ens, case, rank, 
                                                                                                            ensemble, num_sensors, U_tot_u_red, S_tot_u_red, V_tot_red = V_tot_red, open_svd=False, lags=lags, forecast=forecast, 
                                                                                                            surface=False,no_input_u_fluc=True)

                #calculate error metrics for this plane and ensemble case

                #RMS 
                u_recons = utilities.convert_3d_to_2d(u_recons_test)
                u_truth = utilities.convert_3d_to_2d(u_fluc_test)

                RMS_recons[j] = utilities.get_RMS(u_recons)
                RMS_true[j] =  utilities.get_RMS(u_truth)


                #Normalized Mean squared error (NMSE)
                print("calc MSE")
                mse = np.mean((u_truth - u_recons) ** 2, axis=(0,1))  # Mean over spatial and time dimensions
                norm_factor = np.mean(u_truth ** 2, axis=(0, 1))
                MSE_z[j] = mse / norm_factor


                #SSIM
                print("calc SSIM")
                ssim_snapshots = [
                    ssim(u_fluc_test[:, :, t], u_recons_test[:, :, t], data_range=u_fluc_test[ :, :, t].max() - u_fluc_test[ :, :, t].min())
                for t in range(num_test_snaps)
                ]
                ssim_values[j] = np.mean(ssim_snapshots)
                print("ssim_vals: ", ssim_values[j])

                #PSNR
                print("calc PSNR")
                psnr_snapshots = [
                psnr(
                    u_fluc_test[:, :, t],
                    u_recons_test[:, :, t],
                    data_range=u_fluc_test[ :, :, t].max() - u_fluc_test[:, :, t].min()
                )
                for t in range(num_test_snaps)
                ]
                psnr_values[j] = np.mean(psnr_snapshots)  # Average over time
                print("PSNR val: ", psnr_values[j])

                #PSD
                print("calc PSD")
                psd_error[j] = power_spectral_density_error(u_fluc_full, u_recons_test, 3, DNS=False, DNS_case=None)
                print("PSD val: ", psd_error[j])

            #save error metrics in dictionary
            err_dict = {
                'RMS_recons' : RMS_recons,
                'RMS_true' : RMS_true,
                'MSE_z' : MSE_z,
                'ssim' : ssim_values,
                'psnr' : psnr_values,
                'psd' : psd_error
            }

            if full_planes:
                plane_string ="_full_planes"
            
            else:
                plane_string = "_planes"
                for i in range(len(vel_planes)):
                    plane_string = plane_string + "_" +  plane
            if forecast:
                fcast ="_forecast_"
            else:
                fcast="_"
            
            err_fname = METRICS_DIR / ("err_metrics_TEE" + fcast +case+"_r"+str(rank)+"_sens"+str(num_sensors)+ "_SHRED_ens"+ str(ensemble)+"_Tee_ens" + str(experimental_ensembles[k]) + plane_string +  ".mat")
            
            with h5py.File(err_fname, 'w') as f:
                for key, value in err_dict.items():
                    f.create_dataset(key, data=value)
            print("saved successfully!")




def calculate_temporal_error_metrics(DNS_case, rank, vel_plane, num_sensors, SHRED_ens, lags=52, forecast=False, full_planes=True):
    """
    Compute time-series error metrics for SHRED reconstructions of a DNS data set.

    The function iterates over one or more previously-trained SHRED ensembles,
    extracts the reconstructed velocity snapshots for the specified test indices,
    and returns per-snapshot metrics that quantify the temporal fidelity of the model.

    ---------- Parameters
    DNS_case : str
        Identifier of the DNS data set ('S1' or 'S2').
    rank : int
        Truncation rank r used for the reduced SVD representation fed to SHRED.
    vel_planes : list[int]
        Indices of the horizontal velocity planes that were stacked into the
        SVD array and reconstructed (surface elevation is **not** included here).
    num_sensors : int
        Number of point sensors at the surface that served as SHRED inputs.
    SHRED_ens : int
        Ensemble index for the SHRED run
    lags : int, default 52
        Length of the sliding-window sequence used during SHRED training;
        needed because the last *lag–1* snapshots are excluded from the index pool.
    forecast : bool, default False
        If *True* the SHRED run was performed in “forecast” mode (continuous
        hold-out block at the end of the time series); otherwise random
        sampling was used.  Affects how ``utilities.open_SHRED`` resolves files.
    full_planes : bool, default True
        If *True* the entire set of DNS planes (57 or 76 depending on case) was
        stacked; if *False* only the planes listed in *vel_planes* are present
        in the SVD stack.  Needed for correct plane indexing.

    ---------- Returns
    RMS_true : np.ndarray
        Array of shape ``(n_snapshots,)`` with the ground-truth planar RMS of
        the target velocity component (computed over horizontal plane but per time
        step).
    RMS_recons : np.ndarray
        Same as RMS_true but for the SHRED reconstruction.
    mse_snapshots : np.ndarray
        Normalised mean-square-error for every test snapshot
        MSE(t) / ||u_truth(t)||².
    ssim_snapshots : np.ndarray
        Structural similarity index between truth and reconstruction,
        computed per snapshot.
    psnr_snapshots : np.ndarray
        Peak signal-to-noise ratio (dB) per snapshot.
    psd_snapshots : np.ndarray
        Normalised power-spectral-density error per snapshot, as returned by
        ``power_spectral_density_error_time_series``.
        Length equals ``len(test_indices)``.

    ---------- Notes.
    * At present the function only *returns* the metrics for the **last** loop
      iteration (last SHRED ensemble & last plane).  Extend the code or wrap
      this function if you need full ensemble/plane arrays.

    """


    #useful to define total number of planes per case
    #helps for picking out correct planes if we only want to
    #plot some (but not all) planes
    DNS_case = utilities.case_name_converter(DNS_case)
    
    #open and reduce SVD
    stack_planes=[vel_plane] 

    U_tot_red, S_tot_red, V_tot_red = utilities.stack_svd_arrays_DNS(stack_planes, rank, DNS_case=DNS_case)                                                  
    
    #then iterate SHRED ensembles
    print("start ensemble looping")

    V_tot_recons, V_tot_svd, test_indices = utilities.open_SHRED(None, DNS_case, rank, num_sensors, SHRED_ens, stack_planes, DNS=True, 
                                                                     full_planes=full_planes, forecast=forecast)
        
    num_test_snaps = len(test_indices)
    RMS_recons = np.zeros(len(test_indices))
    RMS_true = np.zeros(len(test_indices))

    vel_plane 
    plane_index = 1 #shift with 1 to compensate for surface elevation when loading V matrices
            

            
    u_fluc=None
    u_fluc_test, u_svd_test, u_recons_test, u_fluc_full = utilities.get_test_imgs_SHRED_DNS(DNS_case, vel_plane, plane_index, u_fluc, V_tot_recons, test_indices, rank, 
                                                                                                    num_sensors,U_tot_red, S_tot_red, V_tot_red, open_svd=False, lags=lags, 
                                                                                                    forecast=False, surface=False, no_input_u_fluc=True)
    #calculate error metrics for this plane and ensemble case

    #RMS 
    u_recons = utilities.convert_3d_to_2d(u_recons_test)
    u_truth = utilities.convert_3d_to_2d(u_fluc_test)

    RMS_recons = utilities.RMS_plane(u_recons)
    RMS_true =  utilities.RMS_plane(u_truth)


    #Mean squared error (MSE)
    #print("calc MSE")
    mse = np.mean((u_truth - u_recons) ** 2, axis=(0))  # Mean over spatial dimension
    norm_factor = np.mean(u_truth ** 2, axis=(0))
    mse_snapshots = mse / norm_factor


    #SSIM
    #print("calc SSIM")
    ssim_snapshots = [
        ssim(u_fluc_test[:, :, t], u_recons_test[:, :, t], data_range=u_fluc_test[ :, :, t].max() - u_fluc_test[ :, :, t].min())
    for t in range(num_test_snaps)
    ]
 

    #PSNR
    #print("calc PSNR")
    psnr_snapshots = [
    psnr(
        u_fluc_test[:, :, t],
        u_recons_test[:, :, t],
        data_range=u_fluc_test[ :, :, t].max() - u_fluc_test[:, :, t].min()
    )
    for t in range(num_test_snaps)
    ]


    #PSD
    #print("calc PSD")
    psd_snapshots = power_spectral_density_error_time_series(u_fluc_full, u_recons_test, 3, test_indices, DNS=True, DNS_case=DNS_case)
    #print("PSD error: ", psd_snapshots)
        
        
    return RMS_true, RMS_recons, mse_snapshots, ssim_snapshots, psnr_snapshots, psd_snapshots 



def calculate_temporal_error_metrics_exp(case, rank, u_fluc, vel_plane, num_sensors, SHRED_ens, experimental_ens, lags=52, forecast=False, full_planes=True):
    """
    Compute per-snapshot error metrics for a SHRED reconstruction of
    experimental velocity data.

    ----------
    Parameters
    ----------
    case
        Experimental case label, e.g. E1 or E2 (passed through
        ``utilities.case_name_converter`` internally).
    rank
        Truncation rank *r* of the reduced SVD used during SHRED training.
    u_fluc
        Optional full-rank velocity field of shape ``(nx, ny, nt)``; if
        ``None`` the field is loaded inside the helper.
    vel_planes
        List of plane indices **(1–5)** to analyse.  Each index is mapped to
        the physical plane name ``['H395','H390','H375','H350','H300']``.
    num_sensors
        Number of surface sensors used by the SHRED model (needed only for
        file-naming conventions).
    SHRED_ens
        Single integer identifying which SHRED ensemble run to evaluate.
    experimental_ens
        int 
    lags
        Length of the sliding window used during SHRED training (default 52).
    forecast
        If *True*, SHRED was trained in “forecast” mode; affects file paths.
    full_planes
        If *False*, SHRED was trained with only the specified plane; if *True*
        the full 5-plane stack was used.  Determines plane indexing.

    ----------
    Returns
    ----------
    RMS_true : np.ndarray
        Ground-truth RMS per snapshot (shape ``(n_test,)``).
    RMS_recons : np.ndarray
        Reconstructed RMS per snapshot.
    mse_snapshots : np.ndarray
        Normalised MSE for each snapshot.
    ssim_snapshots : np.ndarray
        SSIM values for each snapshot.
    psnr_snapshots : np.ndarray
        PSNR values (dB) for each snapshot.
    psd_snapshots : np.ndarray
        Normalised PSD error for each snapshot.


    """

    case = utilities.case_name_converter(case)


    #open and reduce SVD matrices for the given Teetank ensemble
        
       
    plane = vel_plane
    print("plane: ", plane)
    planes = ['H395', 'H390', 'H375', 'H350', 'H300']
    plane = planes[vel_plane-1]
    print("Plane: ", plane)
    X_surf, Y_surf, X_vel, Y_vel = utilities.get_mesh_exp(case, plane)
    #open SHRED for this plane-surface-pairing, experimental ensemble and SHRED ensemble
    V_tot_recons, V_tot_svd, test_indices = utilities.open_SHRED(experimental_ens, case, rank, num_sensors, SHRED_ens, [vel_plane], DNS=False, exp_plane=plane, full_planes=full_planes, forecast=forecast)
    num_test_snaps = len(test_indices)
    #get SVDs correctly
    U_tot_u_red, S_tot_u_red, U_tot_surf_red, S_tot_surf_red, V_tot_red = utilities.open_and_reduce_SVD(experimental_ens, case, rank, forecast=forecast, DNS=False, DNS_plane=None,
                                                                                                       DNS_surf=False, exp=True, plane=plane)


    surf_fluc=None
    u_fluc_test, u_svd_test, u_recons_test, u_fluc_full = utilities.get_test_imgs_SHRED_exp(plane, surf_fluc, u_fluc, V_tot_recons, V_tot_svd, test_indices, X_surf, X_vel, experimental_ens, case, rank, 
                                                                                                SHRED_ens, num_sensors, U_tot_u_red, S_tot_u_red, V_tot_red = V_tot_red, open_svd=False, lags=52, forecast=forecast, 
                                                                                                surface=False,no_input_u_fluc=True)

    #calculate error metrics for this plane and ensemble case

    #RMS 
    u_recons = utilities.convert_3d_to_2d(u_recons_test)
    u_truth = utilities.convert_3d_to_2d(u_fluc_test)

    RMS_recons = utilities.RMS_plane(u_recons)
    RMS_true =  utilities.RMS_plane(u_truth)


    #Mean squared error (MSE)
    #print("calc MSE")
    mse = np.mean((u_truth - u_recons) ** 2, axis=(0))  # Mean over spatial and time dimensions
    norm_factor = np.mean(u_truth ** 2, axis=(0))
    mse_snapshots = mse / norm_factor


    #SSIM
    #print("calc SSIM")
    ssim_snapshots = [
        ssim(u_fluc_test[:, :, t], u_recons_test[:, :, t], data_range=u_fluc_test[ :, :, t].max() - u_fluc_test[ :, :, t].min())
    for t in range(num_test_snaps)
    ]
    #ssim_values = #np.mean(ssim_snapshots)
    #print("ssim_vals: ", ssim_snapshots)

    #PSNR
    #print("calc PSNR")
    psnr_snapshots = [
    psnr(
        u_fluc_test[:, :, t],
        u_recons_test[:, :, t],
        data_range=u_fluc_test[ :, :, t].max() - u_fluc_test[:, :, t].min()
    )
    for t in range(num_test_snaps)
    ]
    #psnr_values[j] = np.mean(psnr_snapshots)  # Average over time
    #print("psnr val: ", psnr_snapshots)

    #PSD
    #print("calc PSD")
    psd_snapshots = power_spectral_density_error_time_series(u_fluc_full, u_recons_test, 3, test_indices, DNS=False, DNS_case=None)
        
           


    
    return RMS_true, RMS_recons, mse_snapshots, ssim_snapshots, psnr_snapshots, psd_snapshots 




def get_ensemble_avg_error_metrics(DNS_case, rank, vel_planes, num_sensors, SHRED_ensembles, forecast=False, full_planes=True):
    '''function that calculates the ensemble averaged error metrics, given specified planes and SHRED ensembles
    returns ensemble averaged values, together with the standard deviation error for those averages'''
    
    num_ens = len(SHRED_ensembles)
    num_tot_planes = len(vel_planes)
    
    RMS_recons_ensembles = np.zeros((num_ens,num_tot_planes))
    RMS_true_ensembles = np.zeros((num_ens,num_tot_planes))
    mse_ensembles = np.zeros((num_ens,num_tot_planes))
    ssim_ensembles = np.zeros((num_ens, num_tot_planes))
    psnr_ensembles = np.zeros((num_ens,num_tot_planes))
    psd_ensembles = np.zeros((num_ens,num_tot_planes))
    


    for ens in range(num_ens):
        ensemble = SHRED_ensembles[ens]

        #find error metric file
        if full_planes:
            plane_string ="_full_planes"

        else:
            plane_string = "_planes"
            for i in range(len(vel_planes)):
                plane_string = plane_string + "_" +  str(vel_planes[i]) 
        if forecast:
            fcast ="_forecast_"
        else:
            fcast ="_"

        if DNS_case=='RE2500': 
            err_fname = METRICS_DIR / ("err_metrics" + fcast +"r"+str(rank)+"_sens"+str(num_sensors)+ "_ens"+ str(ensemble)+ plane_string +  ".mat")
        else:
            err_fname = METRICS_DIR / ("err_metrics_RE1000" + fcast +"r"+str(rank)+"_sens"+str(num_sensors)+ "_ens"+ str(ensemble)+ plane_string +  ".mat")

       
        with h5py.File(err_fname, 'r') as err_dict:
            # List all datasets in the file
            RMS_recons = np.array(err_dict['RMS_recons'])
            RMS_true = np.array(err_dict['RMS_true'])
            mse_values = np.array(err_dict['MSE_z'])
            ssim_values = np.array(err_dict['ssim'])
            psnr_values = np.array(err_dict['psnr'])
            psd_error = np.array(err_dict['psd'])
        RMS_recons_ensembles[ens] = RMS_recons
        RMS_true_ensembles[ens] = RMS_true
        mse_ensembles[ens] = mse_values
        ssim_ensembles[ens] = ssim_values
        psnr_ensembles[ens] = psnr_values
        psd_ensembles[ens] = psd_error
    
    #ensemble averaging
    RMS_recons_avg = np.mean(RMS_recons_ensembles, axis=0)
    RMS_true_avg = np.mean(RMS_true_ensembles, axis=0)
    mse_avg = np.mean(mse_ensembles, axis=0)
    ssim_avg = np.mean(ssim_ensembles, axis=0)
    psnr_avg = np.mean(psnr_ensembles, axis=0)
    psd_avg = np.mean(psd_ensembles, axis=0)

    #calculate standard deviations
    std_RMS_recons = np.std(RMS_recons_ensembles, axis=0)
    std_mse = np.std(mse_ensembles, axis=0)
    std_ssim = np.std(ssim_ensembles, axis=0)
    std_psnr = np.std(psnr_ensembles, axis=0)
    std_psd = np.std(psd_ensembles, axis=0)

    return RMS_recons_avg, RMS_true_avg, mse_avg, ssim_avg, psnr_avg, psd_avg, std_RMS_recons, std_mse, std_ssim, std_psnr, std_psd



def get_ensemble_avg_error_metrics_exp(case, rank, vel_planes, num_sensors, SHRED_ensembles, exp_ensembles, forecast=False, full_planes=True):
    '''function that calculates the ensemble averaged error metrics, given specified planes and SHRED ensembles
    returns ensemble averaged values, together with the standard deviation error for those averages'''
    
    num_SHRED_ens = len(SHRED_ensembles)
    num_experimental_ens = len(exp_ensembles)

    RMS_recons_ensembles = np.zeros((num_experimental_ens, num_SHRED_ens,len(vel_planes)))
    RMS_true_ensembles = np.zeros((num_experimental_ens, num_SHRED_ens,len(vel_planes)))
    mse_ensembles = np.zeros((num_experimental_ens, num_SHRED_ens,len(vel_planes)))
    ssim_ensembles = np.zeros((num_experimental_ens, num_SHRED_ens,len(vel_planes)))
    psnr_ensembles = np.zeros((num_experimental_ens, num_SHRED_ens,len(vel_planes)))
    psd_ensembles = np.zeros((num_experimental_ens, num_SHRED_ens,len(vel_planes)))
    
    
    for i in range(num_experimental_ens):
        experimental_ens = exp_ensembles[i]

        for j in range(num_SHRED_ens):
            SHRED_ens = SHRED_ensembles[j]

            #find error metric file
            if full_planes:
                plane_string ="_full_planes"

            else:
                plane_string = "_planes"
                for i in range(len(vel_planes)):
                    depths = ['H395', 'H390', 'H375', 'H350', 'H300']
                    plane_string = plane_string + "_" +  depths[vel_planes[i]-1]
            if forecast:
                fcast ="_forecast_"
            else:
                fcast ="_"

            err_fname = METRICS_DIR / ("err_metrics_TEE" + fcast + case + "_r"+str(rank)+"_sens"+str(num_sensors)+ "_SHRED_ens"+ str(SHRED_ens)+"_Tee_ens" + str(experimental_ens) + plane_string +  ".mat")

            
            with h5py.File(err_fname, 'r') as err_dict:
                # List all datasets in the file
                #print("Keys in the HDF5 file:", list(err_dict.keys()))

                RMS_recons = np.array(err_dict['RMS_recons'])
                RMS_true = np.array(err_dict['RMS_true'])
                mse_values = np.array(err_dict['MSE_z'])
                ssim_values = np.array(err_dict['ssim'])
                psnr_values = np.array(err_dict['psnr'])
                psd_error = np.array(err_dict['psd'])
            RMS_recons_ensembles[i,j] = RMS_recons
            RMS_true_ensembles[i,j] = RMS_true
            mse_ensembles[i,j] = mse_values
            ssim_ensembles[i,j] = ssim_values
            psnr_ensembles[i,j] = psnr_values
            psd_ensembles[i,j] = psd_error
    
    #SHRED ensemble averaging
    RMS_recons_avg = np.mean(RMS_recons_ensembles, axis=1)
    RMS_true_avg = np.mean(RMS_true_ensembles, axis=1)
    mse_avg = np.mean(mse_ensembles, axis=1)
    ssim_avg = np.mean(ssim_ensembles, axis=1)
    psnr_avg = np.mean(psnr_ensembles, axis=1)
    psd_avg = np.mean(psd_ensembles, axis=1)

    #Teetank ensemble averaging
    RMS_recons_avg = np.mean(RMS_recons_avg, axis=0)
    RMS_true_avg = np.mean(RMS_true_avg, axis=0)
    mse_avg = np.mean(mse_avg, axis=0)
    ssim_avg = np.mean(ssim_avg, axis=0)
    psnr_avg = np.mean(psnr_avg, axis=0)
    psd_avg = np.mean(psd_avg, axis=0)

    #calculate standard deviations
    std_RMS_recons = np.std(RMS_recons_ensembles, axis=1)
    std_mse = np.std(mse_ensembles, axis=1)
    std_ssim = np.std(ssim_ensembles, axis=1)
    std_psnr = np.std(psnr_ensembles, axis=1)
    std_psd = np.std(psd_ensembles, axis=1)

    #calculate std in ensemble cases
    std_RMS_recons = np.sqrt(np.sum(np.power(std_RMS_recons,2),axis=0))

    return RMS_recons_avg, RMS_true_avg, mse_avg, ssim_avg, psnr_avg, psd_avg, std_RMS_recons, std_mse, std_ssim, std_psnr, std_psd




def calc_RMS_profile_true(DNS_case, vel_planes, dimT):
    """
    Compute and save vertical RMS profiles time series for selected DNS planes.

    For each plane in `vel_planes`, this computes the planar RMS of u'(x,y,t)
    at every time snapshot, producing an array of shape (n_planes, dimT).
    Results are also written to an HDF5 .mat file on disk
    (filename depends on `DNS_case`).

    Parameters
    ----------
    DNS_case : str
        DNS case identifier (e.g., 'S1', 'S2').
    vel_planes : Sequence[int]
        Indices of horizontal velocity planes to process.
    dimT : int
        Number of time snapshots.

    Returns
    -------
    np.ndarray
        RMS time series with shape (len(vel_planes), dimT).

    """
    DNS_case = utilities.case_name_converter(DNS_case)
    rms_time = np.zeros((len(vel_planes),dimT))

    for i in range(len(vel_planes)):
        plane = i+1
        print("plane: ", plane)
        u_fluc = utilities.get_velocity_plane_DNS(DNS_case,plane)
        u_fluc_2d = utilities.convert_3d_to_2d(u_fluc)
        rms_time[i,:] = utilities.RMS_plane(u_fluc_2d)
    
    rms_dict = {
        'rms_z' : rms_time
    }
    
    if DNS_case=='RE2500':
        rms_fname = METRICS_DIR / ("rms_z_true_RE2500.mat")
    else:
        rms_fname = METRICS_DIR / ("rms_z_true_RE1000.mat")

    with h5py.File(rms_fname, 'w') as f:
        for key, value in rms_dict.items():
            f.create_dataset(key, data=value)
    return rms_time



def get_RMS_profile_true(DNS_case, vel_planes):
    '''function to load the vertical profile of the planar RMS velocities (which is calculated once per case)'''
    
    if DNS_case=='RE2500':
        rms_fname = METRICS_DIR / ("rms_z_true_RE2500.mat")
    else:
        rms_fname = METRICS_DIR / ("rms_z_true_RE1000.mat")
    with h5py.File(rms_fname, 'r') as rms_dict:
        RMS_time = np.array(rms_dict['rms_z'])
    RMS_z_full = np.mean(RMS_time,axis=1)
    RMS_z = np.zeros(len(vel_planes))
    for i in range(len(vel_planes)):
        index = vel_planes[i]
        RMS_z[i] = RMS_z_full[index-1]

    return RMS_z



def calc_RMS_profile_true_exp(vel_planes, dimT, case, num_experimental_ensembles):
    '''num_exp_ens: number of experimental ensembles to load'''
    case = utilities.case_name_converter(case)

    rms_time = np.zeros((num_experimental_ensembles, len(vel_planes), dimT))

    for i in range(len(vel_planes)):
                        
        planes = ['H395', 'H390', 'H375', 'H350', 'H300']
        plane = planes[vel_planes[i]-1]


        u = utilities.read_exp_plane(case=case,depth=plane,variable='U0')
        u = u - np.mean(u, axis=3, keepdims=True)

        for j in range(num_experimental_ensembles):
            u_fluc = u[j]
            u_fluc_2d = utilities.convert_3d_to_2d(u_fluc)
            rms_time[j, i,:] = utilities.RMS_plane(u_fluc_2d)
    rms_dict = {
        'rms_z' : rms_time
    }

    
    rms_fname = METRICS_DIR / ("rms_z_true_Tee_" + case +".mat")
    with h5py.File(rms_fname, 'w') as f:
        for key, value in rms_dict.items():
            f.create_dataset(key, data=value)
    return rms_time
        


def get_RMS_profile_true_exp(case, experimental_ens, experimental_ens_avg=False):
    '''function to load the vertical profile of the planar RMS velocities (which is calculated once per case)'''
    
    rms_fname = METRICS_DIR / ("rms_z_true_Tee_" + case +".mat")

    with h5py.File(rms_fname, 'r') as rms_dict:
        RMS_time = np.array(rms_dict['rms_z'])

    RMS_z_full = np.mean(RMS_time,axis=2)
    if experimental_ens_avg:
        RMS_z_full = np.mean(RMS_z_full, axis=0)
    else:
        RMS_z_full = RMS_z_full[experimental_ens-1]
    return RMS_z_full



def calc_avg_error_DNS(DNS_case, r_vals, vel_planes, sensor_vals, SHRED_ensembles, forecast=False, full_planes=False,r_analysis=True):
    '''Calculates average error along the vertical, for a range of rank values or a range of sensor values

    Parameters
    ----------
    DNS_case : str
        DNS identifier ('RE1000', 'RE2500').
    r_vals : Sequence[int]
        Candidate SVD truncation ranks. Used when `r_analysis=True`. If `r_analysis=False`,
        only the first element is used as a fixed rank.
    vel_planes : Sequence[int]
        Indices of horizontal velocity planes included in the vertical averaging.
    sensor_vals : Sequence[int]
        Candidate numbers of surface sensors. Used when `r_analysis=False`. If
        `r_analysis=True`, only the first element is used as a fixed sensor count.
    SHRED_ensembles : Sequence[int]
        SHRED ensemble IDs to average over.
    forecast : bool, optional
        If True, use forecasting split (train first 90%, test last 10%); otherwise random splits.
    full_planes : bool, optional
        If True, treat planes as a contiguous full stack; otherwise use only `vel_planes`.
    r_analysis : bool, optional
        If True, sweep over `r_vals` (rank analysis) with fixed sensor count.
        If False, sweep over `sensor_vals` (sensor analysis) with fixed rank.


    Returns:
    mse_list : list of MSE averaged over SHRED ensembles and vel_planes, with length of var_vals
    ssim_list : list of SSIM averaged over SHRED ensembles and vel_planes, with length of var_vals
    psnr_list : list of PSNR averaged over SHRED ensembles and vel_planes, with length of var_vals 
    psd_list : list of psd errors averaged over SHRED ensembles and vel_planes, with length of var_vals
    '''
    
    #can either study a range of rank values (r_vals) or a range of sensor values (sensor_vals)
    if r_analysis:
        var_vals = r_vals
        num_sensors = sensor_vals[0]
    else:
        var_vals = sensor_vals
        rank = r_vals[0]
    
    mse_list = np.zeros(len(var_vals))
    ssim_list = np.zeros(len(var_vals))
    psnr_list = np.zeros(len(var_vals))
    psd_list = np.zeros(len(var_vals))
    
    for i in range(len(var_vals)):
       
        if r_analysis:
            rank = r_vals[i]
        else:
            num_sensors=sensor_vals[i]

        RMS_recons_avg, RMS_true_avg, mse_avg, ssim_avg, psnr_avg, psd_avg, std_RMS_recons, std_mse_z, std_ssim, std_psnr, std_psd= get_ensemble_avg_error_metrics(
            DNS_case,rank, vel_planes, num_sensors, SHRED_ensembles, forecast=forecast, full_planes=full_planes)
        mse_list[i] = np.mean(mse_avg)
        ssim_list[i] = np.mean(ssim_avg)
        psnr_list[i] = np.mean(psnr_avg)
        psd_list[i] = np.mean(psd_avg)

    return mse_list, ssim_list, psnr_list, psd_list



def calc_avg_error_exp(case, r_vals, vel_planes, sensor_vals, SHRED_ensembles, exp_ensembles, forecast=False, r_analysis=True):
    '''Calculates average error along the vertical, for a range of rank values or a range of sensor values
        for the experimental case

    Parameters
    ----------
    case : str
        experimental identifier ('P25', 'P50').
    r_vals : Sequence[int]
        Candidate SVD truncation ranks. Used when `r_analysis=True`. If `r_analysis=False`,
        only the first element is used as a fixed rank.
    vel_planes : Sequence[int]
        Indices of horizontal velocity planes included in the vertical averaging.
    sensor_vals : Sequence[int]
        Candidate numbers of surface sensors. Used when `r_analysis=False`. If
        `r_analysis=True`, only the first element is used as a fixed sensor count.
    SHRED_ensembles : Sequence[int]
        SHRED ensemble IDs to average over.
    exp_ensembles : Sequence[int]
        experimental ensemble cases to average over.
    forecast : bool, optional
        If True, use forecasting split (train first 90%, test last 10%); otherwise random splits.
    full_planes : bool, optional
        If True, treat planes as a contiguous full stack; otherwise use only `vel_planes`.
    r_analysis : bool, optional
        If True, sweep over `r_vals` (rank analysis) with fixed sensor count.
        If False, sweep over `sensor_vals` (sensor analysis) with fixed rank.


    Returns:
    mse_list : list of MSE averaged over SHRED ensembles and vel_planes, with length of var_vals
    ssim_list : list of SSIM averaged over SHRED ensembles and vel_planes, with length of var_vals
    psnr_list : list of PSNR averaged over SHRED ensembles and vel_planes, with length of var_vals 
    psd_list : list of psd errors averaged over SHRED ensembles and vel_planes, with length of var_vals
    '''
    
    
    mse_list = np.zeros(len(r_vals))
    ssim_list = np.zeros(len(r_vals))
    psnr_list = np.zeros(len(r_vals))
    psd_list = np.zeros(len(r_vals))
    
    if r_analysis:
        var_vals = r_vals
        num_sensors = sensor_vals[0]
    else:
        var_vals = sensor_vals
        rank = r_vals[0]
    
    mse_list = np.zeros(len(var_vals))
    ssim_list = np.zeros(len(var_vals))
    psnr_list = np.zeros(len(var_vals))
    psd_list = np.zeros(len(var_vals))
    
    for i in range(len(var_vals)):
       
        if r_analysis:
            rank = r_vals[i]
        else:
            num_sensors=sensor_vals[i]


        rank = r_vals[i]
        RMS_recons_avg, RMS_true_avg, mse_avg, ssim_avg, psnr_avg, psd_avg, std_RMS_recons, std_mse_z, std_ssim, std_psnr, std_psd= get_ensemble_avg_error_metrics_exp(case, 
                                rank, vel_planes, num_sensors, SHRED_ensembles, exp_ensembles, forecast, full_planes=True)
        
        mse_list[i] = np.mean(mse_avg)
        ssim_list[i] = np.mean(ssim_avg)
        psnr_list[i] = np.mean(psnr_avg)
        psd_list[i] = np.mean(psd_avg)

    return mse_list, ssim_list, psnr_list, psd_list




    ###### SHRED codes

