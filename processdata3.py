import os
import scipy.integrate
import torch
from scipy.io import loadmat
import numpy as np
import scipy.linalg
import utilities3 as utilities
import mat73
import h5py
from lvpyio import read_set
from sklearn.preprocessing import MinMaxScaler
import models
import torch
import matplotlib.pyplot as plt

class TimeSeriesDataset(torch.utils.data.Dataset):
    '''Takes input sequence of sensor measurements with shape (batch size, lags, num_sensors)
    and corresponding measurments of high-dimensional state, return Torch dataset'''
    #Q: what is meant by "lags"?
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.len = X.shape[0]
        
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
    def __len__(self):
        return self.len


def stack_svd_arrays(vel_planes, rank, DNS=True, DNS_case='RE2500', teetank_ens=None, teetank_case=None, tee_forecast=False):
    '''Load SVD matrices for selected planes + surface elevation
        and stack them in the shape of
        [U1, U2, U3,...Un]
        and likewise for S and V matrices'''
    
    r=1000
    num_planes = len(vel_planes)
    
    if DNS:
        #extract lowest velocity plane
        DNS_plane = vel_planes[-1]
        U_tot_red, S_tot_red, V_tot_red= utilities.open_and_reduce_SVD(teetank_ens, teetank_case, r, rank, forecast=False, DNS_new=True, 
            DNS_plane=DNS_plane, DNS_surf=False, DNS_case=DNS_case, Teetank=False)
        
        V_tot = V_tot_red
        U_tot = U_tot_red
        S_tot = S_tot_red

        #iterate planes from lower to upper and stack their U, S, V matrices on top of each other
        for plane in range(num_planes-2,-1,-1):
            DNS_plane = vel_planes[plane]
            U, S, V = utilities.open_and_reduce_SVD(teetank_ens, teetank_case, r, rank, forecast=False, DNS_new=True, DNS_plane=DNS_plane, DNS_surf=False, DNS_case=DNS_case, Teetank=False)
            
            U_tot = np.hstack((U, U_tot))
            S_tot = np.hstack((S, S_tot))
            V_tot = np.hstack((V,V_tot))
        
        #extract surface elevation SVD and stack on top of the velocity U, S, V matrices
        U_surf, S_surf, V_surf = utilities.open_and_reduce_SVD(teetank_ens, teetank_case, r, rank, forecast=False, DNS_new=True, DNS_plane=None, DNS_surf=True, DNS_case=DNS_case, Teetank=False)
        U_tot = np.hstack((U_surf, U_tot))
        S_tot = np.hstack((S_surf, S_tot))
        V_tot = np.hstack((V_surf,V_tot))

    else:
        #not properly implemented for experimental data, but might be unnecessary here anyway
        U_tot_u_red, S_tot_u_red, U_tot_eta_red, S_tot_eta_red, V_tot_red = utilities.open_and_reduce_SVD(teetank_ens, teetank_case, r, rank, forecast=tee_forecast, DNS_new=False, DNS_plane=None, DNS_surf=False, Teetank=True)
    
    return U_tot, S_tot, V_tot


#done
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
        b. Reconstruct the velocity field at rank ``r_new``.  
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
    * Integrates with `utilities.compute_psd_1d` for PSD calculation.
    * The first row of `psd_multi` corresponds to the full‑rank PSD.
    """

    dimX, dimY, dimT = utilities.get_dims_DNS(DNS_case)
    
    #set spatial resolution based on DNS data
    dx = 2*np.pi/dimX
    dy=2*np.pi/dimY 
    
    u_fluc = np.transpose(u_fluc, (2,0,1))
    PSD_vals, k_vals = utilities.compute_psd_1d(u_fluc, dx=dx, dy=dy, DNS=True)

    integral_length_scale=utilities.get_integral_length_scale(DNS_case)
    
    PSD_vals_all = np.zeros((num_ens, len(PSD_vals)))
    psd_multi_recon = np.zeros((len(rank_list), len(k_vals)))
    psd_multi_recon[0] = PSD_vals
    for i in range(1,len(rank_list)):
        r_new = rank_list[i]
       
        PSD_vals_all = np.zeros((num_ens, len(PSD_vals)))
        for j in range(num_ens):

            U, S, V = utilities.open_and_reduce_SVD(None, None, 1000, r_new, forecast=False, DNS_new=True, DNS_plane=plane, DNS_surf=False, DNS_case=DNS_case, Teetank=False, Tee_plane=None)
            u_svd = U @ np.diag(S) @ np.transpose(V)
            u_svd = utilities.convert_2d_to_3d(u_svd, dimX, dimY, dimT)
            u_svd = np.transpose(u_svd, (2,0,1))
            PSD_vals, k_vals = utilities.compute_psd_1d(u_svd, dx=dx, dy=dy, DNS=True)
            PSD_vals_all[j] = PSD_vals

        PSD_avg = np.mean(PSD_vals_all, axis=0)

        psd_multi_recon[i] = PSD_avg
    
    #normalize k axis with integral length scale:
    k_vals = k_vals*integral_length_scale
    spectral_max=np.amax(psd_multi_recon)
    psd_multi = psd_multi_recon/spectral_max
    return psd_multi, k_vals


#done
def calculate_PSD_r_vals_exp(u_fluc, rank_list, case,  r, ensembles, plane):
    """
    Compute ensemble‑averaged 1‑D power‑spectral densities (PSD) for several
    SVD truncation ranks in an experimental Teetank velocity plane.

    The function:
        1. For each desired truncation rank in ``rank_list``  
           a. Loads reduced SVD factors (`U`, `S`, `V`) for each ensemble  
           b. Reconstructs the velocity field at rank ``r_new``  
           c. Calculates a stream‑wise PSD via `utilities.compute_psd_1d`  
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
    dimX, dimY, dimT = utilities.get_dims_teetank_vel()
    u_fluc_nonan = np.nan_to_num(u_fluc[ensemble-1])
    u_fluc_nonan = np.transpose(u_fluc_nonan, (2,0,1))
    PSD_vals, k_vals = utilities.compute_psd_1d(u_fluc_nonan, dx=dx, dy=dy)

    PSD_vals_all = np.zeros((num_ens, len(PSD_vals)))
    psd_multi_recon = np.zeros((len(rank_list), len(k_vals)))

    for i in range(len(rank_list)):
       
        r_new = rank_list[i]
        
        PSD_vals_all = np.zeros((num_ens, len(k_vals)))
        for j in range(len(ensembles)):

            ensemble = ensembles[j]
            
            #get SVD matrices for velocity field
            U_tot_u_red, S_tot_u_red, U_tot_eta_red, S_tot_eta_red, V_tot_red = utilities.open_and_reduce_SVD(ensemble, case, r, r_new, forecast=False, DNS_new=False, DNS_plane=None, DNS_surf=False, DNS_case='RE2500', Teetank=True, Tee_plane=plane)

            u_svd = U_tot_u_red @ np.diag(S_tot_u_red) @ np.transpose(V_tot_red[:,r_new:2*r_new])
            u_svd = utilities.convert_2d_to_3d(u_svd, dimY, dimX, dimT)
            u_fluc_nonan = np.nan_to_num(u_svd)
            u_svd = np.transpose(u_fluc_nonan, (2,0,1))
            #print(u_svd.shape)
            #PSD_vals, k_vals = utilities.compute_psd(u_fluc_nonan, dx=dx, dy=dy)
            PSD_vals, k_vals = utilities.compute_psd_1d(u_svd, dx=dx, dy=dy) #get_PSD_spectrum(u_fluc_nonan, swapaxis=False)
            
            PSD_vals_all[j] = PSD_vals

        PSD_avg = np.mean(PSD_vals_all, axis=0)

        psd_multi_recon[i] = PSD_avg

    spectral_max=np.amax(psd_multi_recon)
    psd_multi = psd_multi_recon/spectral_max

    #normalize k axis with integral length scale:
    k_vals = k_vals*integral_length_scale

    return psd_multi, k_vals




'''SHRED ANALYSIS FUNCTIONS'''
def SHRED_ensemble_DNS(r_vals, num_sensors, ens_start, ens_end, vel_planes, lags, full_planes=True, random_sampling=True, DNS_case='S2', criterion='MSE'):
    """
    Train and evaluate SHRED on multiple DNS ensembles and SVD ranks, then
    save reconstructed test snapshots to `.mat` files.

    Parameters
    ----------
    r_vals : list[int] or ndarray
        SVD truncation ranks to loop over.
    num_sensors : int
        Number of randomly placed surface sensors.
    ens_start, ens_end : int
        Inclusive range of DNS ensemble numbers to process.
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
    * File paths are Windows‑style; adjust `adr_loc` for portability.
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
            U_tot, S_tot, V_tot = stack_svd_arrays(vel_planes, r, DNS_case=DNS_case)
            
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
            m = (load_X).shape[1]   #number of planes * number of SVD modes, plus number of sensors

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
                print("test: ", test_indices)
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
            for i in range(len(all_data_in)): #iterate insert transformed traning data in sequences
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

            train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
            valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
            test_dataset = TimeSeriesDataset(test_data_in, test_data_out)
            

            #DOING SHRED

            shred = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
            validation_errors = models.fit(shred, train_dataset, valid_dataset, criterion='MSE', batch_size=64, num_epochs=3000, lr=1e-3, verbose=True, patience=5)
            print("SHRED successfully done!")
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
            
            #NOTE: hard-coded file naming system
            adr_loc = "C:\\Users\krissmoe\OneDrive - NTNU\PhD\PhD code\PhD-1\Flow Reconstruction and SHRED\MAT_files"
            
            if full_planes:
                plane_string ="_full_planes"

            else:
                plane_string = "_planes"
                for i in range(len(vel_planes)):
                    plane_string = plane_string + "_" +  str(vel_planes[i]) 

            if random_sampling:
                SHRED_fname = adr_loc + "\SHRED_r"+ str(r) +"_" +str(num_sensors) +"sensors_ens" + str(p) + plane_string +".mat"
                if DNS_case=='RE1000':
                    SHRED_fname = adr_loc + "\SHRED_RE1000_r"+ str(r) +"_" +str(num_sensors) +"sensors_ens" + str(p) + plane_string +".mat"
            else:
                if DNS_case=='RE1000':
                    SHRED_fname = adr_loc + "\SHRED_FORECAST_RE1000_r"+ str(r) +"_" +str(num_sensors) +"sensors_ens" + str(p) + plane_string + ".mat"
                else:
                    SHRED_fname = adr_loc + "\SHRED_FORECAST_r"+ str(r) +"_" +str(num_sensors) +"sensors_ens" + str(p) + plane_string + ".mat"
            with h5py.File(SHRED_fname, 'w') as f:
                for key, value in SHRED_dict.items():
                    f.create_dataset(key, data=value)



def SHRED_ensemble_Tee(r_vals, sensors, X, ens_start, ens_end, case, teetank_ens, lags=52, Tee_plane='H390', random_sampling=True, forecast_Tee=False, criterion='MSE'):
    #TODO: figure out what to do with Teetank ensembles
    #ONLY RUN SHRED FOR 1 PLANE PLUS SURFACE AT THE SAME TIME!
    r=900
    U_tot_u, S_tot_u, U_tot_eta, S_tot_eta, V_tot = utilities.open_SVD(r, teetank_ens, vel_fluc=False, variable='u', Teetank=True, teetank_case=case,  DNS_new=False, DNS_plane=None, DNS_surf=False, Tee_plane=Tee_plane)
    print(S_tot_u.shape)
    X_eta, Y_eta, X_vel, Y_vel = utilities.get_mesh_Teetank(case, Tee_plane)
    for p in range(ens_start, ens_end+1):
        #an ensemble
        for q in range(len(r_vals)):
            r = r_vals[q]
            U_tot_u_red, S_tot_u_red, V_tot_red = utilities.reduce_SVD(U_tot_u, S_tot_u, V_tot, levels=2, r_new=r, Tee=True, surf=False)
            print("r: ", r, "\n ens: ", p)
            #U_tot, S_tot, V_tot = open_SVD(r) #load SVD files
            
            
            #do SHRED below
            m2 = r #len(U) # svd modes used
            print(m2)
            num_sensors = sensors
            
            
            nx = X_eta.shape[0] #NOTE: THESE COULD BE SWAPPED!
            ny = X_eta.shape[1]

            load_X = V_tot_red
            n2 = X_eta.shape[0]*X_eta.shape[1]
            #assign random sensor placements
            sensor_locations_ne = np.random.choice(n2, size=num_sensors, replace=False)
            print("sensor_loc: ", sensor_locations_ne)
            sensor_locations = np.arange(0,num_sensors,1, dtype=int)
    
            #want to stack the sensor temporal data on top of the total V transposed array
            load_X = np.hstack((X[sensor_locations_ne,:].T,load_X)) #horizontal stacking of arrays, columnwise, concatenation along 2nd axis
            n = (load_X).shape[0] #12000 snapshots
            m = (load_X).shape[1] #303 total modes

            #creating a mask: grid with zeros expect the sensor points that's assigned with 1
            mask = np.zeros(n2)
            for i in range(num_sensors):
                mask[sensor_locations_ne[i]]=1

            mask2 = mask.reshape((nx,ny))
        
            if random_sampling:
                n = (load_X).shape[0]
                train_indices = np.random.choice(n - lags, size=int(0.8*n), replace=False)
                #create a mask that marks the validation snapshots. value=1 for validation, and zero for training
                mask = np.ones(n - lags) #creates 2000-52 =1948 ones, but we fill in 0s at the train indices
                mask[train_indices] = 0

                valid_test_indices = np.arange(0, n - lags)[np.where(mask!=0)[0]] #indices not used for training
                valid_indices = valid_test_indices[::2] #pick every other index for validation and the others for test
                test_indices = valid_test_indices[1::2]
                print("test: ", test_indices)
                n_test = test_indices.size

                
            else:
                n = (load_X).shape[0]
                n_valid = int(n*0.1)
                n_train = int(n*0.8) #n - lags - n_test - n_valid
                print("n_train: ", n_train)
                n_test = n-lags - n_valid - n_train
                print("n_test: ", n_test)
                #train_indices = np.arange(0, int(n*0.85))
                #create a mask that marks the validation snapshots. value=1 for validation, and zero for training
                #mask = np.ones(n_train + n_valid) 
                #mask[train_indices] = 0
                train_indices = np.arange(0, n_train)
                valid_indices = np.arange(n_train, n_train + n_valid)
                test_indices = np.arange(n_train + n_valid, n - lags)

    

            print("valid_indices shape: ", valid_indices.shape)
            print("test_indices shape: ", test_indices.shape)
            print("train_indices shape: ", train_indices.shape)



            sc = MinMaxScaler()
            sc = sc.fit(load_X[train_indices]) #computes min/max of training data for later scaling
            transformed_X = sc.transform(load_X) #use the previous scaling to fit and transform the training data


            ### Generate input sequences to a SHRED model
            all_data_in = np.zeros((n - lags, lags, num_sensors)) #(2000-52, 52, 3)
            for i in range(len(all_data_in)): #iterate 2000-52 times, and insert transformed traning data in sequences in a gliding way
                all_data_in[i] = transformed_X[i:i+lags, sensor_locations]

            print("transformed data: ", all_data_in)
            ### Generate training validation and test datasets both for reconstruction of states and forecasting sensors
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print("device: ", device)

            train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
            valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
            test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)

            ### -1 to have output be at the same time as final sensor measurements
            train_data_out = torch.tensor(transformed_X[train_indices + lags - 1], dtype=torch.float32).to(device)
            valid_data_out = torch.tensor(transformed_X[valid_indices + lags - 1], dtype=torch.float32).to(device)
            test_data_out = torch.tensor(transformed_X[test_indices + lags - 1], dtype=torch.float32).to(device)

            train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
            valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
            test_dataset = TimeSeriesDataset(test_data_in, test_data_out)
            
            print("test_dataset: ", test_dataset)



            #DOING THE SHRED

            shred = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
            validation_errors = models.fit(shred, train_dataset, valid_dataset, batch_size=64, num_epochs=3000, lr=1e-3, verbose=True, patience=5)
            print("Done")
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
            #Now we test the reconstruction
            n_s = num_sensors
            test_recons = sc.inverse_transform(shred(test_dataset.X).detach().cpu().numpy()) 
            test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy()) 
            print(np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth))

            SHRED_dict = {
                'test_recons': test_recons,
                'test_ground_truth': test_ground_truth,
                'test_indices' : test_indices,
            }
            print(r)
            adr_loc = "C:\\Users\krissmoe\OneDrive - NTNU\PhD\PhD code\PhD-1\Flow Reconstruction and SHRED\MAT_files"
            if random_sampling:
                SHRED_fname = adr_loc + "\Teetank_SHRED_new_ens"+ str(teetank_ens) + "_"+ case + "_" + Tee_plane + "_r"+ str(r) +"_" +str(num_sensors) +"sensors_ens" + str(p) +".mat"
            else:
                SHRED_fname = adr_loc + "\SHRED_r"+ str(r) +"_" + case + "_" +Tee_plane +"_" +str(num_sensors) +"sensors_ens" + str(p) +"_prediction.mat"
            with h5py.File(SHRED_fname, 'w') as f:
                for key, value in SHRED_dict.items():
                    f.create_dataset(key, data=value)

""" The following functions are used to compute depth dependent error metrics of the reconstructed data (RD) when 
compared to ground truth (GT). The eight functions focus on flow statistics (flow rmse, TKE), turbulent statistics 
(autocorrelations, PSD error), and image reconstruction (NMSE, SSIM, PSNR).
1. rms_profile
2. vorticity_profile (NOT YET IMPLEMENTED)
5. Normalized mean square error
6. Power spectral density error
7. Structural similarity index measure
8. Peak signal to noise ratio"""
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.fft import fft2, fftshift


def rms_profile(gt, recon):
    """Compute the depth dependent RMS velocity profile for ground truth and reconstruction."""
    nz, nx, ny, nt = gt.shape
    rms_gt = np.zeros(nz)
    rms_recon = np.zeros(nz)

    for z in range(nz):
        # Compute RMS over planes, then average in time
        rms_gt[z] = np.mean(np.sqrt(np.mean(gt[z, :, :, :] ** 2, axis=(0, 1))))
        rms_recon[z] = np.mean(np.sqrt(np.mean(recon[z, :, :, :] ** 2, axis=(0, 1))))

    return rms_gt, rms_recon


def calculate_instantaneous_rms_profile(DNS_case, SHRED_ens, rank, num_sensors):
    '''compute depth-dependent RMS velocity profile without time averaging'''
    #open and stack SVD planes
    if DNS_case=='RE2500':
        tot_num_planes=76
        
    else:
        tot_num_planes=57
    
    #open and reduce SVD
    print("starting stacking SVD")

    stack_planes = np.arange(1, tot_num_planes+1)

    U_tot_red, S_tot_red, V_tot_red = stack_svd_arrays(stack_planes, rank, DNS=True, DNS_case=DNS_case, teetank_ens=None, teetank_case=None, tee_forecast=False)     

    #open SHRED ensemble
    V_tot_recons, V_tot_svd, test_indices = utilities.open_SHRED(teetank_ens=None, case=None, r=rank, num_sensors=num_sensors, SHRED_ens=SHRED_ens, plane_list=None, DNS=True, 
                                                                     full_planes=True, forecast=False, DNS_case=DNS_case)

    num_test_snaps = len(test_indices)
    #construct velocity fields from SVD
    rms_gt = np.zeros((tot_num_planes, num_test_snaps))
    rms_recons = np.zeros((tot_num_planes, num_test_snaps))

    for j in range(tot_num_planes):
        plane = j+1
        plane_index = plane #shift with 1 to compensate for surface elevation when loading V matrices
        print("plane: ", plane)
        
        u_fluc=None
        u_fluc_test, u_svd_test, u_recons_test, u_fluc_full = utilities.get_test_imgs_SHRED_DNS(DNS_case, plane, plane_index, u_fluc, V_tot_recons, test_indices, 1000, rank, 
                                                                                                    num_sensors,U_tot_red, S_tot_red, V_tot_red, open_svd=False, lags=52, 
                                                                                                    forecast=False, surface=False, no_input_u_fluc=True)
        u_gt = utilities.convert_3d_to_2d(u_fluc_test)
        u_recons = utilities.convert_3d_to_2d(u_recons_test)
        rms_gt[j] = utilities.RMS_plane(u_gt)
        rms_recons[j] = utilities.RMS_plane(u_recons)

    rms_dict = {
        'rms_gt' : rms_gt,
        'rms_recons' : rms_recons
    }
    adr_loc = "C:\\Users\krissmoe\OneDrive - NTNU\PhD\PhD code\PhD-1\Flow Reconstruction and SHRED\MAT_files"
    if DNS_case=='RE2500':
        rms_fname = adr_loc + "\\rms_rms_SHRED"+ str(SHRED_ens) + "_RE2500.mat"
    else:
        rms_fname = adr_loc + "\\rms_SHRED"+ str(SHRED_ens) + "_RE1000.mat"

    with h5py.File(rms_fname, 'w') as f:
        for key, value in rms_dict.items():
            f.create_dataset(key, data=value)
    
    return rms_gt, rms_recons


def calculate_instantaneous_rms_profile_teetank(teetank_case, teetank_ens, SHRED_ens, rank, num_sensors):
    '''compute depth-dependent RMS velocity profile without time averaging'''
    #open and stack SVD planes

    tot_num_planes=4
    vel_planes = [1,2,3,4,5]
    V_tot_recons, V_tot_svd, test_indices = utilities.open_SHRED(teetank_ens, teetank_case, rank, num_sensors, SHRED_ens, vel_planes, DNS=False,  Tee_plane='H390', full_planes=True, forecast=False)
    num_test_snaps = len(test_indices)

    #construct velocity fields from SVD
    rms_gt = np.zeros((tot_num_planes, num_test_snaps))
    rms_recons = np.zeros((tot_num_planes, num_test_snaps))

    for j in range(tot_num_planes):
        plane_index = j+2

        print("plane: ", plane_index)
        planes = ['H395', 'H390', 'H375', 'H350', 'H300']
        plane = planes[plane_index-1]
        print("plane: ", plane)
        
        X_eta, Y_eta, X_vel, Y_vel = utilities.get_mesh_Teetank(teetank_case, plane)


        #open SHRED for this plane-surface-pairing, Tee-ensemble and SHRED ensemble
        V_tot_recons, V_tot_svd, test_indices = utilities.open_SHRED(teetank_ens, teetank_case, rank, num_sensors, SHRED_ens, vel_planes, DNS=False,  Tee_plane=plane, full_planes=True, forecast=False)
        num_test_snaps = len(test_indices)
        #get SVDs correctly
        U_tot_u_red, S_tot_u_red, U_tot_eta_red, S_tot_eta_red, V_tot_red = utilities.open_and_reduce_SVD(teetank_ens, teetank_case, 900, rank, forecast=False, DNS_new=False, DNS_plane=None,
                                                                                                                   DNS_surf=False, Teetank=True, Tee_plane=plane)

        #at this point, I need my SVD matrices
        eta_fluc=None
        u_fluc=None
        u_fluc_test, u_svd_test, u_recons_test, u_fluc_full = utilities.get_test_imgs_SHRED_Teetank(plane, eta_fluc, u_fluc, V_tot_recons, V_tot_svd, test_indices, X_eta, X_vel, teetank_ens, teetank_case,900, rank, 
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
    adr_loc = "C:\\Users\krissmoe\OneDrive - NTNU\PhD\PhD code\PhD-1\Flow Reconstruction and SHRED\MAT_files"
    
    rms_fname = adr_loc + "\\rms_SHRED_"+teetank_case + "_SHREDens_" +  str(SHRED_ens) + "_teeEns_" + str(teetank_ens) + ".mat"

    with h5py.File(rms_fname, 'w') as f:
        for key, value in rms_dict.items():
            f.create_dataset(key, data=value)
    
    return rms_gt, rms_recons


def open_instantaneous_rms_profile(DNS_case, SHRED_ens, rank, num_sensors):
    '''opens the instantaneous (non-time-averaged) rms profile for a specific DNS case and SHRED ensemble'''
    
    adr_loc = "C:\\Users\krissmoe\OneDrive - NTNU\PhD\PhD code\PhD-1\Flow Reconstruction and SHRED\MAT_files"
    if DNS_case=='RE2500':
        rms_fname = adr_loc + "\\rms_rms_SHRED"+ str(SHRED_ens) + "_RE2500.mat"
    else:
        rms_fname = adr_loc + "\\rms_SHRED"+ str(SHRED_ens) + "_RE1000.mat"
    with h5py.File(rms_fname, 'r') as rms_dict:
        rms_gt = np.array(rms_dict['rms_gt'])
        rms_recons = np.array(rms_dict['rms_recons'])

    return rms_gt, rms_recons

def open_instantaneous_rms_profile_teetank(teetank_case, teetank_ens, SHRED_ens, rank, num_sensors):
    '''opens the instantaneous (non-time-averaged) rms profile for a specific DNS case and SHRED ensemble'''
    
    adr_loc = "C:\\Users\krissmoe\OneDrive - NTNU\PhD\PhD code\PhD-1\Flow Reconstruction and SHRED\MAT_files"
    rms_fname = adr_loc + "\\rms_SHRED_"+teetank_case + "_SHREDens_" +  str(SHRED_ens) + "_teeEns_" + str(teetank_ens) + ".mat"
    with h5py.File(rms_fname, 'r') as rms_dict:
        rms_gt = np.array(rms_dict['rms_gt'])
        rms_recons = np.array(rms_dict['rms_recons'])

    return rms_gt, rms_recons

def normalized_mean_square_error(gt, recon):
    """Compute normalized mean square error (NMSE) between ground truth and reconstruction."""
    mse = np.mean((gt - recon) ** 2, axis=(1, 2, 3))  # Mean over spatial and time dimensions
    norm_factor = np.mean(gt ** 2, axis=(1, 2, 3))  # Normalization factor
    return mse / norm_factor



def power_spectral_density_error_v2(gt, recon, num_scales, DNS=True, DNS_case='RE2500'):
    """Compute normalized mean error in the power spectral density for the X largest scales."""

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
    # Compute 2D FFT for each (nx, ny) plane and average over time
    #gt_fft = np.mean([np.abs(fft2(gt[:, :, t]))**2 for t in range(nt2)], axis=0)
    #recon_fft = np.mean([np.abs(fft2(recon[:, :, t]))**2 for t in range(nt)], axis=0)

    
            
    
           
            
    gt = np.transpose(gt, (2,0,1))
    gt_fft, k_vals = utilities.compute_psd_1d(gt, dx, dy, DNS)
    recon = np.transpose(recon, (2,0,1))
    recon_fft, k_vals = utilities.compute_psd_1d(recon, dx, dy, DNS)
    # Radially bin the FFT results by wavenumber magnitude
    #kx = np.fft.fftfreq(nx)
    #ky = np.fft.fftfreq(ny)
    #kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    #k_magnitude = np.sqrt(kx_grid**2 + ky_grid**2)

    # Bin indices
    k_bins = np.linspace(0, k_vals.max(), bins)
    psd_diff = []

    #calculate difference along total length of spectrum
    #num_scales=len(recon_fft)
    #for i in range(num_scales):
    #    mask = (k_bins[i] <= k_vals) & (k_vals < k_bins[i + 1])
    #    psd_diff.append(np.mean(np.abs(gt_fft[mask] - recon_fft[mask])))
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
    # Compute 2D FFT for each (nx, ny) plane and average over time
    #gt_fft = np.mean([np.abs(fft2(gt[:, :, t]))**2 for t in range(nt2)], axis=0)
    #recon_fft = np.mean([np.abs(fft2(recon[:, :, t]))**2 for t in range(nt)], axis=0)

    
            
    
           
            
    gt = np.transpose(gt, (2,0,1))
    gt_fft, k_vals = utilities.compute_psd_1d(gt, dx, dy, DNS)
    recon = np.transpose(recon, (2,0,1))
    recon_fft, k_vals = utilities.compute_psd_1d(recon, dx, dy, DNS)
    # Radially bin the FFT results by wavenumber magnitude
    #kx = np.fft.fftfreq(nx)
    #ky = np.fft.fftfreq(ny)
    #kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    #k_magnitude = np.sqrt(kx_grid**2 + ky_grid**2)

    # Bin indices
    k_bins = np.linspace(0, k_vals.max(), bins)
    psd_diff = []
    
    #calculate difference along total length of spectrum
    #num_scales=len(recon_fft)
    #for i in range(num_scales):
    #    mask = (k_bins[i] <= k_vals) & (k_vals < k_bins[i + 1])
    #    psd_diff.append(np.mean(np.abs(gt_fft[mask] - recon_fft[mask])))
    #gt_int = scipy.integrate.simpson(gt_fft[:cutoff_index], x=k_vals[:cutoff_index])
    
    #recon_int = scipy.integrate.simpson(recon_fft[:cutoff_index], x=k_vals[:cutoff_index])
    #psd_error = np.abs(gt_int - recon_int)/gt_int
    # Normalize error by ground truth PSD
    #psd_error = np.sum(psd_diff) / np.sum(gt_fft)

    return gt_fft, recon_fft, k_vals

def power_spectral_density_error_time_series(gt, recon, num_scales, test_indices, DNS=True, DNS_case='RE2500'):
    """Compute normalized mean error in the power spectral density for the X largest scales."""

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
    # Compute 2D FFT for each (nx, ny) plane and average over time
    #gt_fft = np.mean([np.abs(fft2(gt[:, :, t]))**2 for t in range(nt2)], axis=0)
    #recon_fft = np.mean([np.abs(fft2(recon[:, :, t]))**2 for t in range(nt)], axis=0)
           

    gt = np.transpose(gt, (2,0,1))
    gt=gt[test_indices]
    gt_fft, k_vals = utilities.compute_psd_1d(gt, dx, dy, DNS, time_avg=False)
    recon = np.transpose(recon, (2,0,1))
    recon_fft, k_vals = utilities.compute_psd_1d(recon, dx, dy, DNS, time_avg=False)
    num_snaps=gt_fft.shape[0]
    print("shape fft: ", gt_fft.shape)
    # Radially bin the FFT results by wavenumber magnitude
    #kx = np.fft.fftfreq(nx)
    #ky = np.fft.fftfreq(ny)
    #kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    #k_magnitude = np.sqrt(kx_grid**2 + ky_grid**2)

    # Bin indices
    k_bins = np.linspace(0, k_vals.max(), bins)
    psd_diff = []

    #calculate difference along total length of spectrum
    #num_scales=len(recon_fft)
    #for i in range(num_scales):
    #    mask = (k_bins[i] <= k_vals) & (k_vals < k_bins[i + 1])
    #    psd_diff.append(np.mean(np.abs(gt_fft[mask] - recon_fft[mask])))
    #gt_int = np.zeros(num_snaps)
    psd_error = np.zeros(num_snaps)
    for i in range(num_snaps):
        gt_int = scipy.integrate.simpson(gt_fft[i,:cutoff_index], x=k_vals[:cutoff_index])
    
        recon_int = scipy.integrate.simpson(recon_fft[i,:cutoff_index], x=k_vals[:cutoff_index])
        psd_error[i] = np.abs(gt_int - recon_int)/gt_int
    # Normalize error by ground truth PSD
    #psd_error = np.sum(psd_diff) / np.sum(gt_fft)

    return psd_error


def ssim_per_depth(gt, recon):
    """Compute SSIM for each depth layer, averaged over time."""
    nz, nx, ny, nt = gt.shape
    ssim_values = np.zeros(nz)

    for z in range(nz):
        ssim_snapshots = [
            ssim(gt[z, :, :, t], recon[z, :, :, t], data_range=gt[z, :, :, t].max() - gt[z, :, :, t].min())
            for t in range(nt)
        ]
        ssim_values[z] = np.mean(ssim_snapshots)  # Average over time

    return ssim_values



def psnr_per_depth(gt, recon):
    """Compute PSNR for each depth layer, averaged over time."""
    nz, nx, ny, nt = gt.shape
    psnr_values = np.zeros(nz)

    for z in range(nz):
        psnr_snapshots = [
            psnr(
                gt[z, :, :, t],
                recon[z, :, :, t],
                data_range=gt[z, :, :, t].max() - gt[z, :, :, t].min()
            )
            for t in range(nt)
        ]
        psnr_values[z] = np.mean(psnr_snapshots)  # Average over time

    return psnr_values


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


def calculate_error_metrics(DNS_case, r_new, vel_planes, num_sensors, SHRED_ensembles, lags=52, forecast=False, add_surface=False, full_planes=True, new_naming=True):
    r=1000
    teetank_ens=None
    case=None

    #useful to define total number of planes per case
    #helps for picking out correct planes if we only want to
    #plot some (but not all) planes
    if DNS_case=='RE2500':
        tot_num_planes=76
    else:
        tot_num_planes=57
    
    #open and reduce SVD
    print("starting stacking SVD")
    if full_planes:
        stack_planes = np.arange(1, tot_num_planes+1)
    else:
        stack_planes=vel_planes 

    U_tot_red, S_tot_red, V_tot_red = stack_svd_arrays(stack_planes, r_new, DNS=True, DNS_case=DNS_case, teetank_ens=None, teetank_case=None, tee_forecast=False)                                                  
    print("end stacking SVD")
    #then iterate SHRED ensembles

    print("start ensemble looping")
    RMS_ensembles_recons = np.zeros((len(SHRED_ensembles), len(vel_planes)))
    RMS_ensembles_truth = np.zeros((len(SHRED_ensembles), len(vel_planes)))
    for i in range(len(SHRED_ensembles)):
        ensemble = SHRED_ensembles[i]
        print("ensemble: ", ensemble)

        print("open SHRED: ")
        V_tot_recons, V_tot_svd, test_indices = utilities.open_SHRED(teetank_ens, case, r_new, num_sensors, ensemble, vel_planes, DNS=True, 
                                                                     full_planes=full_planes, forecast=forecast, DNS_case=DNS_case)
        
        num_test_snaps = len(test_indices)
        RMS_recons = np.zeros(len(vel_planes))
        RMS_true = np.zeros((len(vel_planes)))

        MSE_z = np.zeros(len(vel_planes))
        ssim_values = np.zeros(len(vel_planes))
        psnr_values = np.zeros(len(vel_planes))
        psd_error = np.zeros(len(vel_planes))

        for j in range(len(vel_planes)):
            plane = vel_planes[j]
            if full_planes and len(vel_planes) < tot_num_planes:
                plane_index = vel_planes[j]
            else:
                plane_index = j+1 #shift with 1 to compensate for surface elevation when loading V matrices
            
            print("plane: ", plane)
            
            u_fluc=None
            u_fluc_test, u_svd_test, u_recons_test, u_fluc_full = utilities.get_test_imgs_SHRED_DNS(DNS_case, plane, plane_index, u_fluc, V_tot_recons, test_indices, r, r_new, 
                                                                                                    num_sensors,U_tot_red, S_tot_red, V_tot_red, open_svd=False, lags=lags, 
                                                                                                    forecast=False, surface=False, no_input_u_fluc=True)

            #calculate error metrics for this plane and ensemble case

            #RMS 
            u_recons = utilities.convert_3d_to_2d(u_recons_test)
            u_truth = utilities.convert_3d_to_2d(u_fluc_test)

            RMS_recons[j] = utilities.get_RMS(u_recons)
            RMS_true[j] =  utilities.get_RMS(u_truth)


            #Mean squared error (MSE)
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
            print("psnr val: ", psnr_values[j])

            #PSD
            print("calc PSD")
            psd_error[j] = power_spectral_density_error_v2(u_fluc_full, u_recons_test, 3, DNS=True, DNS_case=DNS_case)
            print("PSD error: ", psd_error[j])
        err_dict = {
            'RMS_recons' : RMS_recons,
            'RMS_true' : RMS_true,
            'MSE_z' : MSE_z,
            'ssim' : ssim_values,
            'psnr' : psnr_values,
            'psd' : psd_error
        }


        adr_loc = "C:\\Users\krissmoe\OneDrive - NTNU\PhD\PhD code\PhD-1\Flow Reconstruction and SHRED\MAT_files\err_metrics"
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
        if new_naming:
            if DNS_case=='RE2500': 
                err_fname = adr_loc + fcast +"r"+str(r_new)+"_sens"+str(num_sensors)+ "_ens"+ str(ensemble)+ plane_string +  ".mat"
            else:
                err_fname = adr_loc + "_RE1000" + fcast +"r"+str(r_new)+"_sens"+str(num_sensors)+ "_ens"+ str(ensemble)+ plane_string +  ".mat"
        else:
            err_fname = adr_loc + fcast + "ens"+ str(ensemble)+ plane_string +  ".mat"
        with h5py.File(err_fname, 'w') as f:
            for key, value in err_dict.items():
                f.create_dataset(key, data=value)
        print("saved successfully!")

    #take the ensemble avg
    #RMS_z_recons = np.mean(RMS_ensembles_recons, axis=0)
    #RMS_z_truth = np.mean(RMS_ensembles_truth, axis=0)

    #std_RMS_z_recons = np.std(RMS_ensembles_recons, axis=0)
    #std_RMS_z_truth = np.std(RMS_ensembles_truth, axis=0)

    print("DONE!")


    
    #return RMS_z_recons, RMS_z_truth, std_RMS_z_recons, std_RMS_z_truth



def calculate_error_metrics_tee(case, r_new, u_fluc, vel_planes, num_sensors, SHRED_ensembles, Tee_ensembles, lags=52, forecast=False, add_surface=False, full_planes=True, new_naming=False):
    '''error vertically for one Tee ensemble case and one SHRED ensemble case at a time'''
    '''Tee_ensembles: list is ensemble indices
    SHRED_ensembles: list of SHRED ensemble indices'''
    r=900
    #teetank_ens=None
    #case=None


 

    print("start ensemble looping")
    #TODO! Add Teetank ensembles as an extra dimension to save!
    RMS_ensembles_recons = np.zeros((len(SHRED_ensembles), len(vel_planes)))
    RMS_ensembles_truth = np.zeros((len(SHRED_ensembles), len(vel_planes)))
    
    for k in range(len(Tee_ensembles)):
        
        teetank_ens = Tee_ensembles[k]
        #open and reduce SVD matrices for the given Teetank ensemble
        
        RMS_ensembles_recons = np.zeros((len(SHRED_ensembles), len(vel_planes)))
        RMS_ensembles_truth = np.zeros((len(SHRED_ensembles), len(vel_planes)))
        #continue below when I've fixed the above issue
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
                X_eta, Y_eta, X_vel, Y_vel = utilities.get_mesh_Teetank(case, plane)
                #open SHRED for this plane-surface-pairing, Tee-ensemble and SHRED ensemble
                V_tot_recons, V_tot_svd, test_indices = utilities.open_SHRED(teetank_ens, case, r_new, num_sensors, ensemble, vel_planes, DNS=False,  Tee_plane=plane, full_planes=full_planes, forecast=forecast)
                num_test_snaps = len(test_indices)
                #get SVDs correctly
                U_tot_u_red, S_tot_u_red, U_tot_eta_red, S_tot_eta_red, V_tot_red = utilities.open_and_reduce_SVD(teetank_ens, case, r, r_new, forecast=forecast, DNS_new=False, DNS_plane=None,
                                                                                                                   DNS_surf=False, Teetank=True, Tee_plane=plane)

                #at this point, I need my SVD matrices
                eta_fluc=None
                u_fluc_test, u_svd_test, u_recons_test, u_fluc_full = utilities.get_test_imgs_SHRED_Teetank(plane, eta_fluc, u_fluc, V_tot_recons, V_tot_svd, test_indices, X_eta, X_vel, teetank_ens, case,r, r_new, 
                                                                                                            ensemble, num_sensors, U_tot_u_red, S_tot_u_red, V_tot_red = V_tot_red, open_svd=False, lags=52, forecast=forecast, 
                                                                                                            surface=False,no_input_u_fluc=True)

                #calculate error metrics for this plane and ensemble case

                #RMS 
                u_recons = utilities.convert_3d_to_2d(u_recons_test)
                u_truth = utilities.convert_3d_to_2d(u_fluc_test)

                RMS_recons[j] = utilities.get_RMS(u_recons)
                RMS_true[j] =  utilities.get_RMS(u_truth)


                #Mean squared error (MSE)
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
                print("psnr val: ", psnr_values[j])

                #PSD
                print("calc PSD")
                psd_error[j] = power_spectral_density_error_v2(u_fluc_full, u_recons_test, 3, DNS=False, DNS_case=None)
        
            err_dict = {
                'RMS_recons' : RMS_recons,
                'RMS_true' : RMS_true,
                'MSE_z' : MSE_z,
                'ssim' : ssim_values,
                'psnr' : psnr_values,
                'psd' : psd_error
            }


            adr_loc = "C:\\Users\krissmoe\OneDrive - NTNU\PhD\PhD code\PhD-1\Flow Reconstruction and SHRED\MAT_files\err_metrics_TEE"
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
            if new_naming: 
                err_fname = adr_loc + fcast +case+"_r"+str(r_new)+"_sens"+str(num_sensors)+ "_SHRED_ens"+ str(ensemble)+"_Tee_ens" + str(Tee_ensembles[k]) + plane_string +  ".mat"
            else:
                err_fname = adr_loc + fcast + case + "_ens"+ str(ensemble)+ plane_string +  ".mat"
            with h5py.File(err_fname, 'w') as f:
                for key, value in err_dict.items():
                    f.create_dataset(key, data=value)
            print("saved successfully!")

    #take the ensemble avg
    #RMS_z_recons = np.mean(RMS_ensembles_recons, axis=0)
    #RMS_z_truth = np.mean(RMS_ensembles_truth, axis=0)

    #std_RMS_z_recons = np.std(RMS_ensembles_recons, axis=0)
    #std_RMS_z_truth = np.std(RMS_ensembles_truth, axis=0)

    print("DONE!")


    
    #return RMS_z_recons, RMS_z_truth, std_RMS_z_recons, std_RMS_z_truth




def calculate_temporal_error_metrics(DNS_case, r_new, vel_planes, num_sensors, SHRED_ensembles, lags=52, forecast=False, add_surface=False, full_planes=True, new_naming=True):
    r=1000
    teetank_ens=None
    case=None

    #useful to define total number of planes per case
    #helps for picking out correct planes if we only want to
    #plot some (but not all) planes

    
    #open and reduce SVD
    print("starting stacking SVD")


    stack_planes=vel_planes 

    U_tot_red, S_tot_red, V_tot_red = stack_svd_arrays(stack_planes, r_new, DNS=True, DNS_case=DNS_case, teetank_ens=None, teetank_case=None, tee_forecast=False)                                                  
    print("end stacking SVD")
    #then iterate SHRED ensembles

    print("start ensemble looping")
    RMS_ensembles_recons = np.zeros((len(SHRED_ensembles), len(vel_planes)))
    RMS_ensembles_truth = np.zeros((len(SHRED_ensembles), len(vel_planes)))
    for i in range(len(SHRED_ensembles)):
        ensemble = SHRED_ensembles[i]
        print("ensemble: ", ensemble)

        print("open SHRED: ")
        V_tot_recons, V_tot_svd, test_indices = utilities.open_SHRED(teetank_ens, case, r_new, num_sensors, ensemble, vel_planes, DNS=True, 
                                                                     full_planes=full_planes, forecast=forecast, DNS_case=DNS_case)
        
        num_test_snaps = len(test_indices)
        RMS_recons = np.zeros(len(test_indices))
        RMS_true = np.zeros(len(test_indices))

        MSE_z = np.zeros(len(test_indices))
        ssim_values = np.zeros(len(test_indices))
        psnr_values = np.zeros(len(test_indices))
        psd_error = np.zeros(len(test_indices))

        for j in range(len(vel_planes)):
            plane = vel_planes[j]
            plane_index = j+1 #shift with 1 to compensate for surface elevation when loading V matrices
            
            print("plane: ", plane)
            
            u_fluc=None
            u_fluc_test, u_svd_test, u_recons_test, u_fluc_full = utilities.get_test_imgs_SHRED_DNS(DNS_case, plane, plane_index, u_fluc, V_tot_recons, test_indices, r, r_new, 
                                                                                                    num_sensors,U_tot_red, S_tot_red, V_tot_red, open_svd=False, lags=lags, 
                                                                                                    forecast=False, surface=False, no_input_u_fluc=True)

            #calculate error metrics for this plane and ensemble case

            #RMS 
            u_recons = utilities.convert_3d_to_2d(u_recons_test)
            u_truth = utilities.convert_3d_to_2d(u_fluc_test)

            RMS_recons = utilities.RMS_plane(u_recons)
            RMS_true =  utilities.RMS_plane(u_truth)


            #Mean squared error (MSE)
            print("calc MSE")
            mse = np.mean((u_truth - u_recons) ** 2, axis=(0))  # Mean over spatial dimension
            norm_factor = np.mean(u_truth ** 2, axis=(0))
            mse_snapshots = mse / norm_factor


            #SSIM
            print("calc SSIM")
            ssim_snapshots = [
                ssim(u_fluc_test[:, :, t], u_recons_test[:, :, t], data_range=u_fluc_test[ :, :, t].max() - u_fluc_test[ :, :, t].min())
            for t in range(num_test_snaps)
            ]
            ssim_values = ssim_snapshots
            print("ssim_vals: ", ssim_values)

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
            psnr_values = psnr_snapshots  
            print("psnr val: ", psnr_values)

            #PSD
            print("calc PSD")
            psd_snapshots = power_spectral_density_error_time_series(u_fluc_full, u_recons_test, 3, test_indices, DNS=True, DNS_case=DNS_case)
            print("PSD error: ", psd_error[j])
        
        
    return RMS_true, RMS_recons, mse_snapshots, ssim_snapshots, psnr_snapshots, psd_snapshots 



def calculate_temporal_error_metrics_tee(case, r_new, u_fluc, vel_planes, num_sensors, SHRED_ensembles, Tee_ensembles, lags=52, forecast=False, add_surface=False, full_planes=True, new_naming=False):
    '''error vertically for one Tee ensemble case and one SHRED ensemble case at a time'''
    '''Tee_ensembles: list is ensemble indices
    SHRED_ensembles: list of SHRED ensemble indices'''
    r=900
    #teetank_ens=None
    #case=None


 

    print("start ensemble looping")
    #TODO! Add Teetank ensembles as an extra dimension to save!
    RMS_ensembles_recons = np.zeros((len(SHRED_ensembles), len(vel_planes)))
    RMS_ensembles_truth = np.zeros((len(SHRED_ensembles), len(vel_planes)))
    
    for k in range(len(Tee_ensembles)):
        
        teetank_ens = Tee_ensembles[k]
        #open and reduce SVD matrices for the given Teetank ensemble
        
        RMS_ensembles_recons = np.zeros((len(SHRED_ensembles), len(vel_planes)))
        RMS_ensembles_truth = np.zeros((len(SHRED_ensembles), len(vel_planes)))
        #continue below when I've fixed the above issue
        for i in range(len(SHRED_ensembles)):
            ensemble = SHRED_ensembles[i]
            print("ensemble: ", ensemble)


            for j in range(len(vel_planes)):
                plane = vel_planes[j]
                print("plane: ", plane)
                planes = ['H395', 'H390', 'H375', 'H350', 'H300']
                plane = planes[vel_planes[j]-1]
                print("Plane: ", plane)
                X_eta, Y_eta, X_vel, Y_vel = utilities.get_mesh_Teetank(case, plane)
                #open SHRED for this plane-surface-pairing, Tee-ensemble and SHRED ensemble
                V_tot_recons, V_tot_svd, test_indices = utilities.open_SHRED(teetank_ens, case, r_new, num_sensors, ensemble, vel_planes, DNS=False,  Tee_plane=plane, full_planes=full_planes, forecast=forecast)
                num_test_snaps = len(test_indices)
                #get SVDs correctly
                U_tot_u_red, S_tot_u_red, U_tot_eta_red, S_tot_eta_red, V_tot_red = utilities.open_and_reduce_SVD(teetank_ens, case, r, r_new, forecast=forecast, DNS_new=False, DNS_plane=None,
                                                                                                                   DNS_surf=False, Teetank=True, Tee_plane=plane)

                #at this point, I need my SVD matrices
                eta_fluc=None
                u_fluc_test, u_svd_test, u_recons_test, u_fluc_full = utilities.get_test_imgs_SHRED_Teetank(plane, eta_fluc, u_fluc, V_tot_recons, V_tot_svd, test_indices, X_eta, X_vel, teetank_ens, case,r, r_new, 
                                                                                                            ensemble, num_sensors, U_tot_u_red, S_tot_u_red, V_tot_red = V_tot_red, open_svd=False, lags=52, forecast=forecast, 
                                                                                                            surface=False,no_input_u_fluc=True)

                #calculate error metrics for this plane and ensemble case

                #RMS 
                u_recons = utilities.convert_3d_to_2d(u_recons_test)
                u_truth = utilities.convert_3d_to_2d(u_fluc_test)

                RMS_recons = utilities.RMS_plane(u_recons)
                RMS_true =  utilities.RMS_plane(u_truth)


                #Mean squared error (MSE)
                print("calc MSE")
                mse = np.mean((u_truth - u_recons) ** 2, axis=(0))  # Mean over spatial and time dimensions
                norm_factor = np.mean(u_truth ** 2, axis=(0))
                mse_snapshots = mse / norm_factor


                #SSIM
                print("calc SSIM")
                ssim_snapshots = [
                    ssim(u_fluc_test[:, :, t], u_recons_test[:, :, t], data_range=u_fluc_test[ :, :, t].max() - u_fluc_test[ :, :, t].min())
                for t in range(num_test_snaps)
                ]
                #ssim_values = #np.mean(ssim_snapshots)
                print("ssim_vals: ", ssim_snapshots)

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
                #psnr_values[j] = np.mean(psnr_snapshots)  # Average over time
                print("psnr val: ", psnr_snapshots)

                #PSD
                print("calc PSD")
                psd_snapshots = power_spectral_density_error_time_series(u_fluc_full, u_recons_test, 3, test_indices, DNS=False, DNS_case=None)
        
           


    
    return RMS_true, RMS_recons, mse_snapshots, ssim_snapshots, psnr_snapshots, psd_snapshots 




def get_ensemble_avg_error_metrics(DNS_case,r_new, vel_planes, num_sensors, SHRED_ensembles, forecast=False, full_planes=True, new_naming=True):
    '''function that calculates the ensemble averaged error metrics, given specified planes and SHRED ensembles
    returns ensemble averaged values, together with the standard deviation error for those averages'''
    
    num_ens = len(SHRED_ensembles)

    adr_loc = "C:\\Users\krissmoe\OneDrive - NTNU\PhD\PhD code\PhD-1\Flow Reconstruction and SHRED\MAT_files\err_metrics"

    if DNS_case=='RE2500' and full_planes:
        num_tot_planes = 76
    elif DNS_case=='RE1000' and full_planes:
        num_tot_planes=57
    else:
        num_tot_planes = len(vel_planes)
    RMS_recons_ensembles = np.zeros((num_ens,num_tot_planes))
    RMS_true_ensembles = np.zeros((num_ens,num_tot_planes))
    mse_z_ensembles = np.zeros((num_ens,num_tot_planes))
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
        if new_naming:
            if DNS_case=='RE2500': 
                err_fname = adr_loc + fcast +"r"+str(r_new)+"_sens"+str(num_sensors)+ "_ens"+ str(ensemble)+ plane_string +  ".mat"
            else:
                err_fname = adr_loc + "_RE1000" + fcast +"r"+str(r_new)+"_sens"+str(num_sensors)+ "_ens"+ str(ensemble)+ plane_string +  ".mat"
        else:
            err_fname = adr_loc + fcast +"ens"+ str(ensemble)+ plane_string +  ".mat"
        
        with h5py.File(err_fname, 'r') as err_dict:
            # List all datasets in the file
            RMS_recons = np.array(err_dict['RMS_recons'])
            RMS_true = np.array(err_dict['RMS_true'])
            mse_z = np.array(err_dict['MSE_z'])
            ssim_values = np.array(err_dict['ssim'])
            psnr_values = np.array(err_dict['psnr'])
            psd_error = np.array(err_dict['psd'])
        RMS_recons_ensembles[ens] = RMS_recons
        RMS_true_ensembles[ens] = RMS_true
        mse_z_ensembles[ens] = mse_z
        ssim_ensembles[ens] = ssim_values
        psnr_ensembles[ens] = psnr_values
        psd_ensembles[ens] = psd_error
    
    #ensemble averaging
    RMS_recons_avg = np.mean(RMS_recons_ensembles, axis=0)
    RMS_true_avg = np.mean(RMS_true_ensembles, axis=0)
    mse_avg = np.mean(mse_z_ensembles, axis=0)
    ssim_avg = np.mean(ssim_ensembles, axis=0)
    psnr_avg = np.mean(psnr_ensembles, axis=0)
    psd_avg = np.mean(psd_ensembles, axis=0)

    #calculate standard deviations
    std_RMS_recons = np.std(RMS_recons_ensembles, axis=0)
    std_mse_z = np.std(mse_z_ensembles, axis=0)
    std_ssim = np.std(ssim_ensembles, axis=0)
    std_psnr = np.std(psnr_ensembles, axis=0)
    std_psd = np.std(psd_ensembles, axis=0)

    return RMS_recons_avg, RMS_true_avg, mse_avg, ssim_avg, psnr_avg, psd_avg, std_RMS_recons, std_mse_z, std_ssim, std_psnr, std_psd



def get_ensemble_avg_error_metrics_Tee(teetank_case, r_new, vel_planes, num_sensors, SHRED_ensembles, Tee_ensembles, lags=52, forecast=False, add_surface=False, full_planes=True, new_naming=False):
    '''function that calculates the ensemble averaged error metrics, given specified planes and SHRED ensembles
    returns ensemble averaged values, together with the standard deviation error for those averages'''
    
    num_SHRED_ens = len(SHRED_ensembles)
    num_teetank_ens = len(Tee_ensembles)

    adr_loc = "C:\\Users\krissmoe\OneDrive - NTNU\PhD\PhD code\PhD-1\Flow Reconstruction and SHRED\MAT_files\err_metrics_TEE"

    RMS_recons_ensembles = np.zeros((num_teetank_ens, num_SHRED_ens,len(vel_planes)))
    RMS_true_ensembles = np.zeros((num_teetank_ens, num_SHRED_ens,len(vel_planes)))
    mse_z_ensembles = np.zeros((num_teetank_ens, num_SHRED_ens,len(vel_planes)))
    ssim_ensembles = np.zeros((num_teetank_ens, num_SHRED_ens,len(vel_planes)))
    psnr_ensembles = np.zeros((num_teetank_ens, num_SHRED_ens,len(vel_planes)))
    psd_ensembles = np.zeros((num_teetank_ens, num_SHRED_ens,len(vel_planes)))
    
    
    for i in range(num_teetank_ens):
        tee_ens = Tee_ensembles[i]

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
            if new_naming:
                err_fname = adr_loc + fcast + teetank_case + "_r"+str(r_new)+"_sens"+str(num_sensors)+ "_SHRED_ens"+ str(SHRED_ens)+"_Tee_ens" + str(tee_ens) + plane_string +  ".mat"
            else:
                err_fname = adr_loc + fcast + teetank_case +  "_ens"+ str(tee_ens)+ plane_string +  ".mat"
        
            with h5py.File(err_fname, 'r') as err_dict:
                # List all datasets in the file
                #print("Keys in the HDF5 file:", list(err_dict.keys()))

                RMS_recons = np.array(err_dict['RMS_recons'])
                RMS_true = np.array(err_dict['RMS_true'])
                mse_z = np.array(err_dict['MSE_z'])
                ssim_values = np.array(err_dict['ssim'])
                psnr_values = np.array(err_dict['psnr'])
                psd_error = np.array(err_dict['psd'])
            RMS_recons_ensembles[i,j] = RMS_recons
            RMS_true_ensembles[i,j] = RMS_true
            mse_z_ensembles[i,j] = mse_z
            ssim_ensembles[i,j] = ssim_values
            psnr_ensembles[i,j] = psnr_values
            psd_ensembles[i,j] = psd_error
    
    #SHRED ensemble averaging
    RMS_recons_avg = np.mean(RMS_recons_ensembles, axis=1)
    RMS_true_avg = np.mean(RMS_true_ensembles, axis=1)
    mse_avg = np.mean(mse_z_ensembles, axis=1)
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
    std_mse_z = np.std(mse_z_ensembles, axis=1)
    std_ssim = np.std(ssim_ensembles, axis=1)
    std_psnr = np.std(psnr_ensembles, axis=1)
    std_psd = np.std(psd_ensembles, axis=1)

    #calculate std in ensemble cases
    std_RMS_recons = np.sqrt(np.sum(np.power(std_RMS_recons,2),axis=0))

    return RMS_recons_avg, RMS_true_avg, mse_avg, ssim_avg, psnr_avg, psd_avg, std_RMS_recons, std_mse_z, std_ssim, std_psnr, std_psd






def calc_RMS_profile_true(DNS_case, vel_planes, dimT):
    '''calculates rms time series for all velocity planes'''
    rms_time = np.zeros((len(vel_planes),dimT))
    print("start")
    for i in range(len(vel_planes)):
        plane = i+1
        print("plane: ", plane)
        u_fluc = utilities.get_velocity_plane_DNS(DNS_case,plane)
        u_fluc_2d = utilities.convert_3d_to_2d(u_fluc)
        rms_time[i,:] = utilities.RMS_plane(u_fluc_2d)
    
    rms_dict = {
        'rms_z' : rms_time
    }
    adr_loc = "C:\\Users\krissmoe\OneDrive - NTNU\PhD\PhD code\PhD-1\Flow Reconstruction and SHRED\MAT_files"
    if DNS_case=='RE2500':
        rms_fname = adr_loc + "\\rms_z_true.mat"
    else:
        rms_fname = adr_loc + "\\rms_z_true_RE1000.mat"

    with h5py.File(rms_fname, 'w') as f:
        for key, value in rms_dict.items():
            f.create_dataset(key, data=value)
    return rms_time



def get_RMS_profile_true(DNS_case, vel_planes):
    '''function to load the vertical profile of the planar RMS velocities (which is calculated once per case)'''
    
    adr_loc = "C:\\Users\krissmoe\OneDrive - NTNU\PhD\PhD code\PhD-1\Flow Reconstruction and SHRED\MAT_files"
    
    if DNS_case=='RE2500':
        rms_fname = adr_loc + "\\rms_z_true.mat"
    else:
        rms_fname = adr_loc + "\\rms_z_true_RE1000.mat"
    with h5py.File(rms_fname, 'r') as rms_dict:
        RMS_time = np.array(rms_dict['rms_z'])
    RMS_z_full = np.mean(RMS_time,axis=1)
    RMS_z = np.zeros(len(vel_planes))
    for i in range(len(vel_planes)):
        index = vel_planes[i]
        RMS_z[i] = RMS_z_full[index-1]

    return RMS_z



def calc_RMS_profile_true_Tee(vel_planes, dimT, Teetank_case, num_teetank_ens):

    rms_time = np.zeros((num_teetank_ens, len(vel_planes), dimT))

    for i in range(len(vel_planes)):
                        
        planes = ['H395', 'H390', 'H375', 'H350', 'H300']
        plane = planes[vel_planes[i]-1]


        u = utilities.read_teetank_plane(case=Teetank_case,depth=plane,variable='U0',surface=False)
        u = u - np.mean(u, axis=3, keepdims=True)

        for j in range(num_teetank_ens):
            u_fluc = u[j]
            u_fluc_2d = utilities.convert_3d_to_2d(u_fluc)
            rms_time[j, i,:] = utilities.RMS_plane(u_fluc_2d)
    rms_dict = {
        'rms_z' : rms_time
    }

    adr_loc = "C:\\Users\krissmoe\OneDrive - NTNU\PhD\PhD code\PhD-1\Flow Reconstruction and SHRED\MAT_files"
    rms_fname = adr_loc + "\\rms_z_true_Tee_" + Teetank_case +".mat"
    with h5py.File(rms_fname, 'w') as f:
        for key, value in rms_dict.items():
            f.create_dataset(key, data=value)
    return rms_time
        


def get_RMS_profile_true_Tee(vel_planes, Teetank_case, Tee_ens, Tee_ens_avg=False):
    adr_loc = "C:\\Users\krissmoe\OneDrive - NTNU\PhD\PhD code\PhD-1\Flow Reconstruction and SHRED\MAT_files"
    rms_fname = adr_loc + "\\rms_z_true_Tee_" + Teetank_case +".mat"

    with h5py.File(rms_fname, 'r') as rms_dict:
        RMS_time = np.array(rms_dict['rms_z'])

    RMS_z_full = np.mean(RMS_time,axis=2)
    if Tee_ens_avg:
        RMS_z_full = np.mean(RMS_z_full, axis=0)
    else:
        RMS_z_full = RMS_z_full[Tee_ens-1]
    return RMS_z_full



def calc_avg_error(DNS_case, r_vals, vel_planes, sensor_vals, SHRED_ensembles, forecast=False, full_planes=False,r_analysis=True):
    '''Calculates average error along the vertical, for a range of rank values or a range of sensor values
    
    Returns:
    mse_list : list of MSE averaged over SHRED ensembles, with length of var_vals
    ssim_list : list of SSIM averaged over SHRED ensembles, with length of var_vals
    psnr_list : list of PSNR averaged over SHRED ensembles, with length of var_vals 
    psd_list : list of psd errors averaged over SHRED ensembles, with length of var_vals
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
            DNS_case,rank, vel_planes, num_sensors, SHRED_ensembles, forecast=forecast, full_planes=full_planes, new_naming=True)
        mse_list[i] = np.mean(mse_avg)
        ssim_list[i] = np.mean(ssim_avg)
        psnr_list[i] = np.mean(psnr_avg)
        psd_list[i] = np.mean(psd_avg)

    return mse_list, ssim_list, psnr_list, psd_list



def calc_avg_error_tee(teetank_case, r_vals, vel_planes, sensor_vals, SHRED_ensembles, Tee_ensembles, forecast=False, add_surface=False, full_planes=False,
    new_naming=True, r_analysis=True):
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


        r_new = r_vals[i]
        RMS_recons_avg, RMS_true_avg, mse_avg, ssim_avg, psnr_avg, psd_avg, std_RMS_recons, std_mse_z, std_ssim, std_psnr, std_psd= get_ensemble_avg_error_metrics_Tee(teetank_case, r_new, vel_planes, num_sensors, SHRED_ensembles, Tee_ensembles, 52, forecast, add_surface, full_planes=True, new_naming=True)
        
        mse_list[i] = np.mean(mse_avg)
        ssim_list[i] = np.mean(ssim_avg)
        psnr_list[i] = np.mean(psnr_avg)
        psd_list[i] = np.mean(psd_avg)

    return mse_list, ssim_list, psnr_list, psd_list




    ###### SHRED codes

