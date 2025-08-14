import scipy as sp
import scipy.io as sio
import numpy as np
import imageio
import h5py
import mat73
from lvpyio import read_set
from skimage.filters import window
from pathlib import Path


'''---------------------------------------------------------------------------------------------------------------------------------'''

'''DATA LOADING AND HANDLING FROM DNS AND 'T-TANK' EXPERIMENTS'''


#done-ish, edit filename
def write_T_tank_PIV_to_mat(case='\P25',depth='\H395',num_ensembles=20, dimX=196, dimY=225, dimT = 900,variable='U0',surface=False, addr=''):
    '''Function to load experimental PIV data from the 'T-tank' turbulent watertank experiment from Davis dataset to MAT-file
    returns MAT-file with velocity field of shape (ens, dimY, dimX, dimT)'''
    
    u_plane = np.zeros((num_ensembles, dimY, dimX, dimT))
    for ens in range(num_ensembles):
        print("ensemble: ", ens)
        src_fname = "F:\\T-Tank" + depth + case + "\Time Resolved" + "\loop=" + str(ens) 
        with read_set(src_fname) as image_set:
            for j in range(dimT):
                image_buffer = image_set[j]
                image_frame = image_buffer[0]
                if image_frame.shape[0] > dimY:
                    im = image_frame[0][variable]
                    u_plane[ens,:,:,j] = im[1:,:]
                else:
                    u_plane[ens,:,:,j] = image_frame[0][variable]
    case = case[1:]
    depth=depth[1:]
    #tee_fname = "E:\\Users\krissmoe\Documents\PhD data storage\T-Tank\case_"+case+"_"+depth+"_"+variable+".mat"
    fname = addr + 'data/exp/raw/case_'+case+"_"+depth+"_"+variable+".mat" 
    PIV_dict = {
        'U': u_plane,
    }

    with h5py.File(fname, 'w') as f:
        for key, value in PIV_dict.items():
            f.create_dataset(key, data=value)
    #sp.io.savemat(svd_fname, svd_dict)
    print("DONE!")

#done-ish, edit filename
def write_T_tank_surf_to_mat(surf_data, case, depth):
    #tee_fname = "E:\\Users\krissmoe\Documents\PhD data storage\T-Tank\case_"+case+"_"+depth+"_"+variable+".mat" 
    tee_fname = "E:\\Users\krissmoe\Documents\PhD data storage\T-Tank\Eta_case_"+case+"_"+depth+".mat" 
    tee_dict = {
        'eta': surf_data,
    }

    with h5py.File(tee_fname, 'w') as f:
        for key, value in tee_dict.items():
            f.create_dataset(key, data=value)
    #sp.io.savemat(svd_fname, svd_dict)
    print("DONE!")


def open_T_tank_profilometry(addr):
    # If address is not provided, ask the user to provide a file path
    if addr is None:
        from tkinter import filedialog
        from tkinter import Tk
        root = Tk()
        root.withdraw()  # Hide the root window
        addr = filedialog.askopenfilename(filetypes=[("Binary files", "*.bin")])
        if not addr:
            return None

    # Open the binary file for reading
    with open(addr, 'rb') as f:
        # Read the size of the array from the first 4 floats
        matsize = np.fromfile(f, dtype=np.float32, count=4)
        matsize = np.flip(matsize)
        print(matsize)
        # Read the rest of the data
        temp = np.fromfile(f, dtype=np.float32)
        print(temp)
    # Reshape the data into the desired 4D shape
    surf_data = temp.reshape(tuple(matsize.astype(int)))
    surf = surf_data.transpose(3, 2, 1, 0)

    return surf


#done
def read_exp_plane(case='P25',depth='H390',variable='U0',addr=''):
    if addr=='':
        fname = 'data/exp/raw/case_'+case+"_"+depth+"_"+variable+".mat" 
    else:
        fname = addr + "case_"+case+"_"+depth+"_"+variable+".mat" 
    
    with h5py.File(fname, 'r') as exp:
        # List all datasets in the file
        #print("Keys in the HDF5 file:", list(tee.keys()))

        U = np.array(exp['U'])
    return U

#done
def read_exp_surface(case='P25', depth='H390', addr=''):
    if addr=='':
        fname = 'data/exp/raw/Eta_case_'+case+"_"+depth + ".mat" 
    else:
        fname = addr + "Eta_case_"+case+"_"+depth+".mat"

    with h5py.File(fname, 'r') as exp:
        # List all datasets in the file
        #print("Keys in the HDF5 file:", list(tee.keys()))

        surf_fluc = np.array(exp['eta'])
    return surf_fluc


#done-ish, edit filename
def get_surface_DNS(DNS_case, addr=''):
    if DNS_case=='RE2500':
        if addr=='':
            fname = "data//DNS/raw/RE2500/surfElev.mat"
        else:
            fname = addr + "surfElev.mat"
    else:

        if addr=='':
            fname = "data//DNS/raw/RE1000/surf_elev.mat"
        else:     
            fname = addr + "surf_elev.mat"
    data = mat73.loadmat(fname)
    if DNS_case=='RE2500':
        surf_full = data['surfElev']
        surf_full = np.transpose(surf_full, (1, 2, 0))
    else:
        surf_full = data['surf_elev']
    surf_mean = np.nanmean(surf_full, axis=2, keepdims=True)

    surf_fluc = surf_full - surf_mean
    return surf_fluc


#done
def get_normalized_surface_DNS(DNS_case):
    '''first reads DNS surface, 
        then normalizes the field'''
    surf_fluc = get_surface_DNS(DNS_case)
    surf_fluc_2d = convert_3d_to_2d(surf_fluc)
    Xnorm= np.max(np.abs(surf_fluc))
    X = surf_fluc_2d/Xnorm #note, this is surface velocity
    print(X.shape)
    #n2 = (X).shape[0]
    return X

#done
def get_normalized_surface_exp(exp_case, plane, experimental_ens):
    '''first reads experimental surface, 
        then normalizes the field'''
    exp_case = case_name_converter(exp_case)
    surf_fluc = get_surface_exp(exp_case, plane)
    surf_fluc_2d = convert_3d_to_2d(surf_fluc[:,:,:,experimental_ens-1])
    Xnorm= np.max(np.abs(surf_fluc))
    X = surf_fluc_2d/Xnorm 
    print(X.shape)

    return X

#done
def get_velocity_plane_DNS(DNS_case, plane, addr=''):
    '''reads velocity plane from the DNS data files
        plane indicates plane index starting from 0'''
    
    if DNS_case == 'RE2500':
        if addr=='':
            fname = "data//DNS/raw/RE2500/" + "u_layer"+str(plane)+".mat"
        else:  
            fname = addr + "u_layer"+str(plane)+".mat"
        dimX = 256
        dimY=256
        dimT=12500
    else:
        if addr=='':
            fname = "data//DNS/raw/RE1000/" + "u_layer"+str(plane)+".mat"
        else:
            fname = addr + "u_layer"+str(plane)+".mat"
        dimX = 128
        dimY=128
        dimT=10900
    data = mat73.loadmat(fname)
    u = data['uPlane']

    #calculates velocity fluctuation, removes mean flow
    u_mean = np.nanmean(u, axis=2, keepdims=True)
    u_fluc = u - u_mean
    return u_fluc

#done
def get_velocity_plane_exp(case, plane, addr=''):
    '''reads PIV velocity planes from experimental case
        with plane indicating plane index, ranging from 1 to 5'''
    depths = ['H395', 'H390', 'H375', 'H350', 'H300']
    depth=depths[plane-1] #plane=1 is H395, plane=2 is H390 etc
    u = read_exp_plane(case,depth,variable='U0', addr=addr)
    #structure of u is (ens, dimY, dimX, dimT)
    u_mean = np.nanmean(u, axis=3, keepdims=True)
    u_fluc = u - u_mean
    return u_fluc

#done
def get_surface_exp(case, plane, addr=''):
    '''reads profilometry surface from experimental case
        for the corresponding paired plane
        with plane indicating plane index, ranging from 1 to 5'''
    depths = ['H395','H390', 'H375', 'H350', 'H300']
    depth=depths[plane-1] #plane=1 is H395, plane=2 is H390 etc
    surf_fluc = read_exp_surface(case, depth, addr)
    return surf_fluc

#done
def get_dims_DNS(DNS_case):
    '''get dimensions for DNS'''
    if DNS_case=='RE2500':
        dimX = 256
        dimY = 256
        dimT = 12500
    else:
        dimX = 128
        dimY = 128
        dimT = 10900
    return dimX, dimY, dimT

#done
def get_dims_exp_vel():
    '''get dimension for PIV fields for the experiment'''
    dimX = 225
    dimY = 196
    dimT = 900
    return dimX, dimY, dimT

#done
def get_dims_exp_surf(depth='H390', case='P50'):
    '''get dimension for profilometry surface field for the experiment'''
    dimX = 610
    dimY = 540
    if depth=='H375' and case=='P50':
        dimY = 543
    dimT = 900
    return dimX, dimY, dimT

#done
def get_mesh_DNS(DNS_case):
    '''get the spatial mesh grid for the DNS'''
    dimX, dimY, dimT = get_dims_DNS(DNS_case)
    X = np.linspace(0,dimX-1, dimX)
    Y = np.linspace(0,dimY-1, dimY) 
    XX, YY = np.meshgrid(X, Y)
    return XX, YY

#done
def get_mesh_exp(case='P50', depth='H390', addr=''):
    '''get the spatial mesh grid for the experiment'''
    addr = addr_string(addr, "data/exp/raw/")
    addr = addr + 'surfMesh_' + depth + '_' + case +'.mat'
    meshes = mat73.loadmat(addr)
    #print(meshes)
    X_surf = meshes['xMesh']
    Y_surf = meshes['yMesh']
    #X_vel = meshes['vMesh']['x']
    #Y_vel = meshes['vMesh']['y']
    dim_vel_X = 196
    dim_vel_Y = 225
    X = np.arange(0,dim_vel_X)
    Y = np.arange(0,dim_vel_Y)
    X_vel, Y_vel = np.meshgrid(X,Y)

    return X_surf, Y_surf, X_vel, Y_vel


#done
def get_zz_DNS(DNS_case, addr=''):
    '''get depth coordinate z for DNS'''
    
    addr = addr_string(addr, 'data/DNS/raw/')
    
    if DNS_case=='RE2500':

        fname = addr + "zz.mat"
    else:
        fname = addr + "zz_RE1000.mat"
    data = sio.loadmat(fname)
    zz = data['zz'][:,0]
    z = 1 -zz
    z = -z*5*np.pi
    return z

#done
def get_zz_exp():
    '''get depth coordinate z for experiment
        values are in cm from surface'''
    z = np.array([-0.5, -1.0, -2.5, -5.0, -10.0])
    return z

#done
def get_integral_length_scale(DNS_case, addr=''):
    '''reads the integral length scale from file'''

    addr = addr_string(addr, 'data/DNS/raw/')
    tscales_fname = addr + "TurbScales_" + DNS_case + ".mat"
    TurbScales = sp.io.loadmat(tscales_fname)

    L_int = TurbScales['TurbScales']['Lint'][0,0][0][0]
    L_visc = TurbScales['TurbScales']['Lvisc'][0,0][0][0]
    L_taylor = TurbScales['TurbScales']['Ltay'][0,0][0][0]
    L_kolmogorov = TurbScales['TurbScales']['Lkolm'][0,0][0][0]
    Re_turb = TurbScales['TurbScales']['ReTur'][0,0][0][0]
    Re_taylor = TurbScales['TurbScales']['ReTay'][0,0][0][0]
    u_Rep = TurbScales['TurbScales']['uRep'][0,0][0][0]
    return L_int

#done
def get_normalized_z(z, z_norm, DNS_case, addr=''):
    '''takes in a z axis (1D array)
        and z_norm argument,
        and normalizes the axis wrt. a length scale
        either None, Taylor length scale, integral scale or mixed

        z_norm: None, 'taylor', 'int', 'mixed'
        '''
    
    #load file with scales:
    addr = addr_string(addr, 'data/DNS/raw/')
    tscales_fname = addr + "TurbScales_" + DNS_case + ".mat"
    TurbScales = sp.io.loadmat(tscales_fname)

    L_int = TurbScales['TurbScales']['Lint'][0,0][0][0]
    L_visc = TurbScales['TurbScales']['Lvisc'][0,0][0][0]
    L_taylor = TurbScales['TurbScales']['Ltay'][0,0][0][0]
    L_kolmogorov = TurbScales['TurbScales']['Lkolm'][0,0][0][0]
    Re_turb = TurbScales['TurbScales']['ReTur'][0,0][0][0]
    Re_taylor = TurbScales['TurbScales']['ReTay'][0,0][0][0]
    u_Rep = TurbScales['TurbScales']['uRep'][0,0][0][0]
    print(L_int) 

    if z_norm=='taylor':
        z = z/L_taylor
    elif z_norm=='int':
        z = z/L_int
    elif z_norm=='mixed':
        z = z/np.sqrt(L_int*L_taylor)
    
    return z

#done
def get_normalized_z_exp(z, z_norm, exp_case):
    '''takes in a z axis (1D array)
        and z_norm argument,
        and normalizes the axis wrt. a length scale
        either None, Taylor length scale, integral scale or mixed

        z_norm: None, 'taylor', 'int', 'mixed'
        '''
    
    #load file with scales:
    #tscales_fname = "E:\\Users\krissmoe\Documents\PhD data storage\SHRED DNS Backup\\TurbScales_" + DNS_case + ".mat"
    #TurbScales = sp.io.loadmat(tscales_fname)
    if exp_case=='P25':

        L_int = 5.1 #cm
        L_visc = None
        L_taylor = None
    elif exp_case=='P50':
        L_int = 6.8 #cm
        L_visc = None
        L_taylor = None
    
    if z_norm=='taylor':
        z = z/L_taylor
        print("NO T-Tank TAYLOR YET!")
    elif z_norm=='int':
        z = z/L_int
    elif z_norm=='mixed':
        z = z/np.sqrt(L_int*L_taylor)
    
    return z

#done
def case_name_converter(case):
    '''converts case name S1, S2, E1, E2 to specific file case names
        necessary for good name flow throughout code base
    '''

    if case=='S1':
        case_out = 'RE1000'
    elif case=='S2':
        case_out = 'RE2500'
    elif case=='E1':
        case_out = 'P25'
    elif case=='E2':
        case_out = 'P50'

    return case_out

'''-------------------------------------------------------------------------------------------------------------------------'''

'''SVD CALCULATIONS AND PRE-PROCESSING SVD MATRICES'''

#done
def save_singular_values_full(DNS_case, DNS_plane):
    '''calculates all singular values for a DNS case
        and saves these to file'''
    u_fluc = get_velocity_plane_DNS(DNS_case, DNS_plane)
    
    print("Starting SVD")
    u_fluc=convert_3d_to_2d(u_fluc)
    U, S, VT = np.linalg.svd(u_fluc,full_matrices=False)
    print(S.shape)
    print("SVD finished")
    del U
    del VT
    S_dict = {
                    'S': S}
    adr_loc = "data/DNS/SVD//"
    S_fname = adr_loc + "S_fullrank_"+ DNS_case + "_plane"+str(DNS_plane)
    with h5py.File(S_fname, 'w') as f:
        for key, value in S_dict.items():
            f.create_dataset(key, data=value)

#done
def save_singular_values_full_exp(case, experimental_ens, plane):
    '''calculates all singular values for an experimental case
        and saves these to file'''
    u_fluc = get_velocity_plane_exp(case, plane)
    u_fluc = u_fluc[experimental_ens-1]
    print("Starting SVD")
    u_fluc=convert_3d_to_2d(u_fluc)
    U, S, VT = np.linalg.svd(u_fluc,full_matrices=False)
    print(S.shape)
    print("SVD finished")
    del U
    del VT
    S_dict = {
                    'S': S}
    adr_loc = "data/exp/SVD//"
    S_fname = adr_loc + "S_fullrank_T_tank_"+ case + "_ens" + str(experimental_ens) +  "_plane"+str(plane)
    with h5py.File(S_fname, 'w') as f:
        for key, value in S_dict.items():
            f.create_dataset(key, data=value)

#done
def get_singular_values_full(DNS_case, DNS_plane):
    '''reads singular values from file, for a DNS case'''
    adr_loc = "data/DNS/SVD//"
    S_fname = adr_loc + "S_fullrank_"+ DNS_case + "_plane"+str(DNS_plane)
    with h5py.File(S_fname, 'r') as s_matrix:
        
        S = np.array(s_matrix['S'])

    return S

#done
def get_singular_values_full_exp(case, experimental_ens, plane):
    '''reads singular values from file, for an experimental case'''
    adr_loc = "data/exp/SVD//"
    S_fname = adr_loc + "S_fullrank_T_tank_"+ case + "_ens" + str(experimental_ens) +  "_plane"+str(plane)
    with h5py.File(S_fname, 'r') as s_matrix:
        
        S = np.array(s_matrix['S'])

    return S

#done
def get_cumsum_svd(r_vals, total_ranks, DNS_case, DNS_plane=10):
    '''calculate cumulative sum of singular values up to rank values 
        given by r_valsfor a DNS case'''
    
    adr_loc = "data/DNS/SVD//"
    S_fname = adr_loc + "S_fullrank_"+ DNS_case + "_plane"+str(DNS_plane)

    with h5py.File(S_fname, 'r') as s_matrix:
        
        S = np.array(s_matrix['S'])
    s_energy = np.zeros(len(r_vals))
    rank_percentage = np.zeros(len(r_vals))
    for i in range(len(r_vals)):
        s_en = np.cumsum(S[:r_vals[i]])/np.sum(S)
        s_energy[i] = s_en[-1]
        rank_percentage[i] = r_vals[i]/total_ranks

    
    return s_energy, rank_percentage


#done
def get_cumsum_svd_exp(r_vals, total_ranks, case, experimental_ens, plane):
    '''calculate cumulative sum of singular values up to rank values 
        given by r_valsfor an experimental case'''
    adr_loc = "data/exp/SVD//"
    S_fname = adr_loc + "S_fullrank_T_tank_"+ case + "_ens" + str(experimental_ens) +  "_plane"+str(plane)
    S_fname = adr_loc + "S_fullrank_T_tank_"+ case + "_ens" + str(experimental_ens) +  "_plane"+str(plane)
    with h5py.File(S_fname, 'r') as s_matrix:
        # List all datasets in the file
        S = np.array(s_matrix['S'])
    s_energy = np.zeros(len(r_vals))
    rank_percentage = np.zeros(len(r_vals))
    for i in range(len(r_vals)):
        s_en = np.cumsum(S[:r_vals[i]])/np.sum(S)
        s_energy[i] = s_en[-1]
        rank_percentage[i] = r_vals[i]/total_ranks
    #s_energy = s_energy
    
    return s_energy, rank_percentage


#done
def save_svd_full(surf_fluc, u_fluc, experimental_ens, exp_case, variable='U', forecast=False, DNS=False, DNS_plane=None, DNS_surf=False, DNS_case='RE2500', new_exp_format=False):
    """
    Compute and save SVD matrices for either DNS (single plane/surface) or
    experimental (per-plane) data, and return the U, S, V matrices.

    Parameters
    ----------
    surf_fluc : (nx*ny, nt) ndarray or None
        Surface-elevation snapshots (flattened space × time). Used when
        DNS_surf=True (DNS branch). Ignored in the experimental branch
        since data are reloaded internally.
    u_fluc : (nx*ny, nt) ndarray or None
        Velocity snapshots (flattened space × time). Used when DNS_surf=False
        (DNS branch). Ignored in the experimental branch.
    experimental_ens : int
        1-based ensemble index for the experimental (Teetank) data.
    exp_case : str
        Experimental case identifier (e.g., 'P25', 'P50').
    variable : str
        Used only for experimental file naming when `new_exp_format=True`.
    forecast : bool, optional
        Kept for naming compatibility; not used in computations.
    DNS : bool, optional
        If True, process DNS data; otherwise process experimental data.
    DNS_plane : int or None
        DNS velocity plane index when `DNS=True` and `DNS_surf=False`.
    DNS_surf : bool, optional
        If True, compute SVD of DNS surface; otherwise DNS velocity plane.
    DNS_case : str, optional
        DNS case identifier (e.g., 'RE2500', 'RE1000').
    new_teetank : bool, optional
        If True, use alternate experimental output filename.

    Returns
    -------
    DNS branch
        (U_tot_u, S_tot_u, V_tot)
            U_tot_u : (nx*ny, r) ndarray
            S_tot_u : (r,) ndarray
            V_tot   : (nt, r) ndarray
        where r=1000 (truncated to the first 1000 singular modes).
    Experimental branch
        (U_tot_u, U_tot_eta, S_tot_u, S_tot_eta, V_tot)
            SVD factors for the **last** processed plane in the loop.


    Notes
    -----
    - Input arrays must be flattened to (nx*ny, nt).
    - Experimental data for each plane are loaded internally; the function
      loops over a predefined set of planes and saves per-plane SVDs.
    """

    if DNS:
        #take in one velocity/surface plane and calculates SVD up to rank 1000
        if DNS_surf:
            del u_fluc
            #assume surf_fluc is the 2d version
            U, S, VT_u = np.linalg.svd(surf_fluc[:,:],full_matrices=False)
        else:
            del surf_fluc
            U, S, VT_u = np.linalg.svd(u_fluc[:,:],full_matrices=False)
        #we only store r=1000 modes
        U_tot_u = U[:, :1000]
        del U
        S_tot_u = S[:1000]
        del S
        V_tot = np.transpose(VT_u[:1000,:])
        del VT_u
    

        
    else:
        #experimental cases

        planes = ['H300', 'H350', 'H375', 'H390']
            
        for i in range(len(planes)):
            plane_str = planes[i]
            u = read_exp_plane(case=exp_case,depth=plane_str,variable='U0')
            surf_fluc = read_exp_surface(case=exp_case, depth=plane_str)
            u_fluc = u - np.mean(u, axis=3, keepdims=True)
            del u
            u_fluc = u_fluc[experimental_ens-1]
            u_fluc = convert_3d_to_2d(u_fluc)
            surf_fluc = surf_fluc[:,:,:,experimental_ens-1]
            surf_fluc = convert_3d_to_2d(surf_fluc)


            #Stack surface elevation on top of one plane at a time!
            #and save this as a separate file!

            U, S, VT = np.linalg.svd(u_fluc,full_matrices=False)
            
            U_tot_u = U
            S_tot_u = S
            V_tot = np.transpose(VT)
        
            U, S, VT = np.linalg.svd(surf_fluc[:,:],full_matrices=False)
            U_tot_surf = U
            S_tot_surf = S
            V_tot = np.hstack((np.transpose(VT),V_tot))
            print("V_tot_dim: ", V_tot.shape)
            
            svd_dict = {
                'U_tot_u': U_tot_u,
                'U_tot_eta': U_tot_surf,
                'S_tot_u': S_tot_u,
                'S_tot_eta': S_tot_surf,
                'V_tot': V_tot}
            adr_loc = "data/exp/SVD"

            svd_fname = adr_loc + "\T_tank SVD_plane_" + plane_str + "_ens"+ str(experimental_ens) + "_"+exp_case + ".mat"
            with h5py.File(svd_fname, 'w') as f:
                for key, value in svd_dict.items():
                    f.create_dataset(key, data=value)
            print("saved successfully!")
                
    print("U shape: ", U_tot_u.shape)
    

    
    #save matrices to mat file
    if DNS:
        adr_loc = "data/exp/SVD"
        svd_dict = {
        'U_tot_u': U_tot_u,
        'S_tot_u': S_tot_u,
        'V_tot': V_tot}

        case_str = DNS_case
        if DNS_surf:
            svd_fname = adr_loc + "\SVD_surf_"+case_str+"_WEinf.mat"
        else:
            svd_fname = adr_loc + "\SVD_plane"+ str(DNS_plane)+ "_"+case_str+"_WEinf.mat"
    elif not new_exp_format:
        svd_dict = {
        'U_tot_u': U_tot_u,
        'U_tot_eta': U_tot_surf,
        'S_tot_u': S_tot_u,
        'S_tot_eta': S_tot_surf,
        'V_tot': V_tot}
    
    else:
        svd_fname = adr_loc + "\T_tank SVD_fullplanes_" + variable + "_ens"+ str(experimental_ens) + "_"+exp_case + ".mat"
    with h5py.File(svd_fname, 'w') as f:
        for key, value in svd_dict.items():
            f.create_dataset(key, data=value)
    #sp.io.savemat(svd_fname, svd_dict)
    print("DONE!")
    if DNS:
        return U_tot_u, S_tot_u, V_tot
    else:
        return U_tot_u, U_tot_surf, S_tot_u, S_tot_surf, V_tot

#done
def calculate_DNS_SVDs(plane_start, plane_end, DNS_case='RE2500', addr=''):
    """
    Compute and save SVD matrices for a range of DNS velocity planes.

    Loads each plane (inclusive range), removes the temporal mean to get
    fluctuations, reshapes to (nx*ny, nt), and calls `save_svd_full` to
    compute a truncated SVD and write .mat files.

    Parameters
    ----------
    plane_start : int
        First DNS plane index to process (inclusive).
    plane_end : int
        Last DNS plane index to process (inclusive).
    DNS_case : str, optional
        DNS case identifier ('RE2500' or 'RE1000'), used for file paths.

    Returns
    -------
    None
        Results are saved to disk via `save_svd_full`.
    """
    
    surf_fluc_2d = np.zeros(1)#utilities.convert_3d_to_2d(u_fluc)
    ens=0
    variable='u'
    case=None
    for plane in range(plane_start, plane_end+1):
        print("plane: ", plane)
      
            
        u_fluc = get_velocity_plane_DNS(DNS_case, plane, addr)


        
        u_fluc = convert_3d_to_2d(u_fluc)
        print("start svd")
        U, S, V = save_svd_full(surf_fluc_2d, u_fluc, ens, case, variable, forecast=False, DNS=True, DNS_plane=plane, DNS_case=DNS_case, DNS_surf=False)
        print("U: ", U.shape)
        print("S: ", S.shape)
        print("V: ", V.shape)
        del U, S, V
    



#done
def open_SVD(experimental_ens, vel_fluc=False, variable='u', exp=False, experimental_case=None, forecast=False, DNS_new=False, DNS_plane=None, DNS_surf=False, DNS_case='RE2500', experimental_plane='H390', addr=''):
    """
    Load precomputed SVD matrices(and optionally raw fluctuations) for DNS or experimental data.

    Parameters
    ----------
    experimental_ens : int
        Experimental ensemble index (used when `exp=True`).
    vel_fluc : bool, optional
        If True, also return the saved velocity fluctuation field `u_fluc` (when available).
    variable : str, optional
        Variable name for experimental files (e.g., 'u').
    exp : bool, optional
        If True, load experimental SVD files; otherwise load DNS.
    experimental_case : str, optional
        Experimental case ID (e.g., 'P25', 'P50') used in filenames (when `exp=True`).
    forecast : bool, optional
        If True, use forecast-style experimental SVD filename pattern.
    DNS_new : bool, optional
        If True, load per-plane DNS SVD (or surface) files using `DNS_plane`/`DNS_surf`.
    DNS_plane : int, optional
        DNS plane index for per-plane loading (used when `DNS_new=True` and `DNS_surf=False`).
    DNS_surf : bool, optional
        If True, load DNS surface SVD file (used when `DNS_new=True`).
    DNS_case : str, optional
        DNS case identifier (e.g., 'RE2500', 'RE1000').
    experimental_plane : str, optional
        Experimental plane identifier (e.g., 'H390').

    Returns
    -------
    tuple
        DNS (default or `DNS_new=False`): (U_tot_u, S_tot_u, V_tot)
        Experimental (`exp=True`): (U_tot_u, S_tot_u, U_tot_eta, S_tot_eta, V_tot)
        If `vel_fluc=True`, an additional `u_fluc` array is appended to the return tuple.
    """
    
    adr_loc = addr_string(addr, "data/DNS/SVD" )
    if exp:
        adr_loc = addr_string(addr, "data/exp/SVD" )
                
        svd_fname = adr_loc + "\T_tank SVD_plane_" + experimental_plane + "_ens"+ str(experimental_ens) + "_"+experimental_case + ".mat"
        if forecast:
            print("finds filename")
            svd_fname = adr_loc + "\T_tank_Forecast_SVD_fullplanes_" + variable + "_"+experimental_case + ".mat"
    elif DNS_new:
        if DNS_surf:
            svd_fname = adr_loc + "\SVD_surf_"+DNS_case+"_WEinf.mat"
        else:
            svd_fname = adr_loc + "\SVD_plane"+ str(DNS_plane) +"_"+DNS_case+"_WEinf.mat"


    #load U, S, V matrices
    with h5py.File(svd_fname, 'r') as SVD:
        U_tot_u = np.array(SVD['U_tot_u']) #U matrix
        S_tot_u = np.array(SVD['S_tot_u'])
        
        if exp:
            #for experimental data, U and S matrices are separated into
            #one set for velocity field (ending with _u)
            #another set for surface elevation (ending with _eta)

            U_tot_surf = np.array(SVD['U_tot_eta']) 
            S_tot_surf = np.array(SVD['S_tot_eta'])
        V_tot = np.array(SVD['V_tot'])
        if vel_fluc:
            u_fluc = np.array(SVD['u_fluc'])
            return U_tot_u, S_tot_u, U_tot_surf, S_tot_surf, V_tot, u_fluc
        else:
            if exp:
                
                return U_tot_u, S_tot_u, U_tot_surf, S_tot_surf, V_tot
            else:
                return U_tot_u, S_tot_u, V_tot


#done
def reduce_SVD(U, S, V, levels, rank, exp=True, DNS_new=False, surf=False):
    '''Function to truncate a high-rank SVD to a lower rank SVD, as given by rank value '''
    if DNS_new:
        levels=1 
    r = V.shape[1]//levels
    
    if not exp:
        U_tot_new = np.zeros((U.shape[0],rank*levels))
        S_tot_new = np.zeros(rank*levels)
    else:
        if surf==True:
            U_tot_new = np.zeros((U.shape[0],rank))
            S_tot_new = np.zeros(rank)
        else:
            
            U_tot_new = np.zeros((U.shape[0],rank))
            
            S_tot_new = np.zeros(rank)
    V_tot_new = np.zeros((V.shape[0],rank*levels))
    for i in range(levels):
        if not exp:
            U_tot_new[:, i*rank:(i+1)*rank] = U[:, i*r:i*r+rank]
            S_tot_new[i*rank:(i+1)*rank] = S[i*r:i*r + rank]
        if surf==False:

            U_tot_new[:, 0:rank] = U[:, 0:rank]
            S_tot_new[0:rank] = S[0:rank]
        V_tot_new[:, i*rank:(i+1)*rank] = V[:, i*r:i*r + rank]

    if exp:
        if surf==True:
            U_tot_new[:,:rank] = U[:, :rank]
            S_tot_new[0:rank] = S[0:rank]

    return U_tot_new, S_tot_new, V_tot_new


#done
def open_and_reduce_SVD(experimental_ens, exp_case, rank, forecast=False, DNS=False, DNS_plane=None, DNS_surf=False, DNS_case='RE2500', exp=True, plane='H390', addr=''):
    """
    Load SVD factors for a selected DNS or experimental plane/surface and
    return rank-truncated matrices

    Parameters
    ----------
    experimental_ens : int
        Experimental ensemble index (ignored for DNS).
    exp_case : str
        Experimental case identifier (e.g., 'P25', 'P50').
    rank : int
        Target truncation rank.
    forecast : bool, optional
        Use forecast-style experimental filenames.
    DNS : bool, optional
        If True, load DNS data; otherwise experimental.
    DNS_plane : int, optional
        DNS plane index (when DNS=True and DNS_surf=False).
    DNS_surf : bool, optional
        If True, load DNS surface SVD instead of a plane.
    DNS_case : str, optional
        DNS case identifier (e.g., 'RE2500').
    exp : bool, optional
        If True, load experimental data.
    plane : str, optional
        Experimental plane (e.g., 'H390').

    Returns
    -------
    tuple
        DNS: (U_u_red, S_u_red, V_red)
        Experimental: (U_u_red, S_u_red, U_eta_red, S_eta_red, V_red)
        where all matrices are truncated to `rank`.
    """
    if DNS:
        U_tot_u, S_tot_u, V_tot = open_SVD(experimental_ens, False, 'u', exp, exp_case, forecast, DNS, DNS_plane, DNS_surf, DNS_case=DNS_case, addr=addr)
        levels=1
    else:
        
        U_tot_u, S_tot_u, U_tot_surf, S_tot_surf, V_tot = open_SVD(experimental_ens, False, 'u', exp,  exp_case, forecast, DNS, DNS_plane, DNS_surf, experimental_plane=plane, addr=addr)
        levels=2
        #extract reduced surface field
        U_tot_surf_red, S_tot_surf_red, V_tot_red = reduce_SVD(U_tot_surf, S_tot_surf, V_tot, levels, rank, exp, DNS, surf=True)
        
    U_tot_u_red, S_tot_u_red, V_tot_red = reduce_SVD(U_tot_u, S_tot_u, V_tot, levels, rank, exp, DNS, surf=False)
    
    if DNS:
        return U_tot_u_red, S_tot_u_red, V_tot_red
    else:
        
        return U_tot_u_red, S_tot_u_red, U_tot_surf_red, S_tot_surf_red, V_tot_red

#done
def stack_svd_arrays_DNS(vel_planes, rank, DNS_case='RE2500', exp_ens=None, exp_case=None, exp_forecast=False):
    '''Load SVD matrices of DNS for selected planes + surface elevation
        and stack them in the shape of
        [U1, U2, U3,...Un]
        and likewise for S and V matrices'''
    
    r=1000
    num_planes = len(vel_planes)
    
    
    #extract lowest velocity plane
    DNS_plane = vel_planes[-1]
    U_tot_red, S_tot_red, V_tot_red= open_and_reduce_SVD(exp_ens, exp_case, rank, forecast=False, DNS=True, 
        DNS_plane=DNS_plane, DNS_surf=False, DNS_case=DNS_case, exp=False)
        
    V_tot = V_tot_red
    U_tot = U_tot_red
    S_tot = S_tot_red

    #iterate planes from lower to upper and stack their U, S, V matrices on top of each other
    for plane in range(num_planes-2,-1,-1):
        DNS_plane = vel_planes[plane]
        U, S, V = open_and_reduce_SVD(exp_ens, exp_case, rank, forecast=False, DNS=True, DNS_plane=DNS_plane, DNS_surf=False, DNS_case=DNS_case, exp=False)
        
        U_tot = np.hstack((U, U_tot))
        S_tot = np.hstack((S, S_tot))
        V_tot = np.hstack((V,V_tot))
        
    #extract surface elevation SVD and stack on top of the velocity U, S, V matrices
    U_surf, S_surf, V_surf = open_and_reduce_SVD(exp_ens, exp_case, rank, forecast=False, DNS=True, DNS_plane=None, DNS_surf=True, DNS_case=DNS_case, exp=False)
    U_tot = np.hstack((U_surf, U_tot))
    S_tot = np.hstack((S_surf, S_tot))
    V_tot = np.hstack((V_surf,V_tot))

    
    return U_tot, S_tot, V_tot


'''--------------------------------------------------------------------------------------------------------------------------------------'''



'''POST-SHRED UTILITY FUNCTIONS FOR SAVING/LOADING FILES, RECONSTRUCTED PLANES ETC'''

#done
def open_SHRED(exp_ens, case, rank, num_sensors, SHRED_ens, plane_list, DNS=True, exp_plane='H390', full_planes=True, forecast=False, addr=''):
    '''loads V matrix for test data from SHRED runs, specified by rank value r, number of sensors (num_sensors) and the SHRED ensemble-case SHRED_ens
        returns 
        test_recons: V matrix for reconstruction 
        test_ground_truth: V Matrix for rank r-compressed test data
        test_indices: the indices in the full dataset where test data is extracted from'''
    
    adr_loc = addr_string(addr, "output/SHRED" )
    
    if DNS:
        if not full_planes:
            plane_string = "_planes"
            for i in range(len(plane_list)):
                plane_string = plane_string + "_" +  str(plane_list[i]) 
        else:
            plane_string ="_full_planes"
        if forecast==False:
            SHRED_fname = adr_loc + "\SHRED_r"+ str(rank) +"_" +str(num_sensors) +"sensors_ens" + str(SHRED_ens) + plane_string +".mat"
            if case=='RE1000':
                SHRED_fname = adr_loc + "\SHRED_RE1000_r"+ str(rank) +"_" +str(num_sensors) +"sensors_ens" + str(SHRED_ens) + plane_string +".mat"
        else:
            SHRED_fname = adr_loc + "\SHRED_FORECAST_r"+ str(rank) +"_" +str(num_sensors) +"sensors_ens" + str(SHRED_ens) + plane_string +".mat"
            if case=='RE1000':
                SHRED_fname = adr_loc + "\SHRED_FORECAST_RE1000_r"+ str(rank) +"_" +str(num_sensors) +"sensors_ens" + str(p) + plane_string + ".mat"
                
    else:
        if forecast:
            print("must change SHRED filename for Teetank forecast")
            SHRED_fname = adr_loc + "\Teetank_FORECAST_case_"+ case + "_r" + str(rank) +"_" +str(num_sensors) +"sensors_ens" + str(SHRED_ens) + ".mat"
        else:
            SHRED_fname = adr_loc + "\Teetank_SHRED_new_ens"+ str(exp_ens) + "_"+ case + "_" + exp_plane + "_r"+ str(rank) +"_" +str(num_sensors) +"sensors_ens" + str(SHRED_ens) +".mat"
    with h5py.File(SHRED_fname, 'r') as SHRED:
        # List all datasets in the file
        #print("Keys in the HDF5 file:", list(SHRED.keys()))

        test_recons = np.array(SHRED['test_recons'])
        test_ground_truth = np.array(SHRED['test_ground_truth'])
        test_indices = np.array(SHRED['test_indices'])

    return test_recons, test_ground_truth, test_indices


#done
def get_test_imgs_SHRED_exp(plane, surf_fluc, u_fluc, V_tot_recons, V_tot_svd, test_indices, X_surf, X_vel, experimental_ens, exp_case, rank, SHRED_ens, num_sensor, U_tot_red=None, S_tot_red=None, V_tot_red = None, open_svd=True, lags=52, forecast=False, surface=False,no_input_u_fluc=False, addr=''):
    """
    Build ground-truth, SVD-truncated, and SHRED-reconstructed test stacks for an
    experimental plane (or the surface) at specified test indices.

    Parameters
    ----------
    plane : str
        Experimental plane label (e.g., 'H390'); if `surface=True`, this refers to surface.
    surf_fluc, u_fluc : np.ndarray or None
        Surface/velocity fluctuation fields. If `no_input_u_fluc=True`, they are loaded internally.
    V_tot_recons : np.ndarray
        SHRED output V-matrix (time coefficients) for reconstruction.
    V_tot_svd : np.ndarray
        SVD V-matrix used in forecast mode.
    test_indices : array-like
        Indices of test snapshots (before lag shift).
    X_surf, X_vel : np.ndarray
        Grids used only for inferring (nx, ny) of surface/velocity fields.
    experimental_ens : int
        Experimental ensemble index to load/select data.
    exp_case : str
        Experimental case identifier (e.g., 'P25', 'P50').
    rank : int
        SVD truncation rank.
    SHRED_ens : int
        SHRED ensemble index (kept for bookkeeping).
    num_sensor : int
        Number of sensors.
    U_tot_red, S_tot_red, V_tot_red : np.ndarray, optional
        Preloaded truncated SVD factors; ignored if `open_svd=True`.
    open_svd : bool, default True
        If True, load and truncate SVD factors internally.
    lags : int, default 52
        Sensor sequence length; shifts test indices by `lags-1`.
    forecast : bool, default False
        Use forecast branch (different V-column layout).
    surface : bool, default False
        If True, operate on surface field instead of subsurface velocity.
    no_input_u_fluc : bool, default False
        If True, load `u_fluc`/`surf_fluc` from disk.

    Returns
    -------
    u_fluc_test : np.ndarray
        Ground-truth test stack, shape (ny, nx, n_test).
    u_svd_test : np.ndarray
        SVD-truncated test stack, shape (ny, nx, n_test).
    u_recons_test : np.ndarray
        SHRED-reconstructed test stack, shape (ny, nx, n_test).
    u_fluc : np.ndarray
        Full fluctuation field used to form `u_fluc_test` (for downstream metrics).
    """        
    
    shift = test_indices + lags - 1

    #if forecast:
    #    dimT=900

    #    #shift indices
    #    shift = shift - dimT*(num_ensembles-1)

    
    num_test_snaps = len(test_indices)
            
    
    if no_input_u_fluc:
        if surface:
            print("load surface")
            surf_fluc = read_exp_surface(case=exp_case, depth=plane)
            u_fluc = surf_fluc[:,:,:,experimental_ens-1]
        else:
            print("load velocity field, plane " + plane)
            
            u =read_exp_plane(case=exp_case,depth=plane,variable='U0')
            
            u_mean = np.nanmean(u, axis=3, keepdims=True)
            u_fluc = u - u_mean
            u_fluc = u_fluc[experimental_ens-1]
        
    u_fluc_test = u_fluc[:,:, shift]
    

    if forecast:
        

        U_u, S_u, U_surf, S_surf, V_tot_red= open_and_reduce_SVD(experimental_ens, exp_case, rank, forecast=True, addr=addr)
    
        #Extract SVD fields
        surf_svd = U_surf@ np.diag(S_surf) @ np.transpose(V_tot_svd[:, 0+num_sensor:rank+num_sensor])
        
        print("surf_svd_shape0: ", surf_svd.shape)
        surf_svd_test = convert_2d_to_3d(surf_svd, X_surf.shape[1], X_surf.shape[0], num_test_snaps)
        del surf_svd
        print("surf_svd_shape: ", surf_svd_test.shape)
        u_svd = U_u @ np.diag(S_u) @ np.transpose(V_tot_svd[:, rank + num_sensor:2*rank+num_sensor])
        
        u_svd_test = convert_2d_to_3d(u_svd, X_vel.shape[1], X_vel.shape[0], num_test_snaps)

        #construct reconstruction
        u_recons_test = U_u @ np.diag(S_u) @ np.transpose(V_tot_recons[:, rank + num_sensor :2*rank + num_sensor])
        del U_u, S_u
        u_recons_test = convert_2d_to_3d(u_recons_test, X_vel.shape[1], X_vel.shape[0], num_test_snaps)

        
        surf_recons_test = U_surf @ np.diag(S_surf) @ np.transpose(V_tot_recons[:, 0 + num_sensor :rank + num_sensor])
        del U_surf, S_surf
        surf_recons_test = convert_2d_to_3d(surf_recons_test, X_surf.shape[1], X_surf.shape[0], num_test_snaps)
    else:
        #Extract SVD fields
        if open_svd:
            U_tot_u_red, S_tot_u_red, U_tot_surf_red, S_tot_surf_red, V_tot_red = open_and_reduce_SVD(experimental_ens, exp_case, rank, forecast=False, DNS=False, DNS_plane=None, DNS_surf=False, exp=True, plane=plane, addr=addr)

            if surface:
                U_tot_red = U_tot_surf_red
                S_tot_red = S_tot_surf_red
            else:
                U_tot_red = U_tot_u_red
                S_tot_red = S_tot_u_red
        
        if surface:
            plane_index=0
            dimY = X_surf.shape[1]
            dimX = X_surf.shape[0]
        else:
            plane_index=1
            dimY = X_vel.shape[1]
            dimX = X_vel.shape[0]
        
        V_tot_red = V_tot_red[:, plane_index*rank :(plane_index+1)*rank]
        V_tot_recons = V_tot_recons[:, plane_index*rank + num_sensor :(plane_index+1)*rank + num_sensor]
        
        #construct svd truncation
        u_svd = U_tot_red @ np.diag(S_tot_red) @ np.transpose(V_tot_red)
        u_svd = u_svd[:,shift]
        u_svd_test = convert_2d_to_3d(u_svd, dimY, dimX, num_test_snaps)

         #construct reconstructions
        u_recons_test = U_tot_red @ np.diag(S_tot_red) @ np.transpose(V_tot_recons) 
        u_recons_test = convert_2d_to_3d(u_recons_test, dimY, dimX, num_test_snaps)

    return u_fluc_test, u_svd_test, u_recons_test, u_fluc


#done
def get_test_imgs_SHRED_DNS(DNS_case, plane, plane_index, u_fluc, V_tot_recons, test_indices, rank, num_sensors, U_tot_red=None, S_tot_red=None, V_tot_red = None, open_svd=True, lags=52, forecast=False, surface=False, no_input_u_fluc=False, addr=''):
    """
    Assemble ground-truth, truncated-SVD, and SHRED-reconstructed velocity
    (or surface-elevation) snapshots for the specified DNS test indices.

    Parameters
    ----------
    DNS_case : str
        DNS identifier (e.g. "S1", "S2").
    plane : int
        Physical plane index used by the loader utilities.
    plane_index : int
        Block index of the velocity plane inside stacked SVD matrices
        (0 = surface, 1 = first velocity plane, …).
    u_fluc : ndarray
        Full-rank fluctuating field (`shape = (ny, nx, nt)`).
        Ignored if `no_input_u_fluc=True`.
    V_tot_recons : ndarray
        Right-singular-vector block returned by SHRED
        (`shape = (nt, n_modes_total)`).
    test_indices : ndarray
        Indices of test snapshots in *original* time base.
    rank : int
        Truncation rank).
    num_sensors : int
        Number of surface sensors prepended to the V–matrix.
    U_tot_red, S_tot_red, V_tot_red : ndarrays, optional
        Pre-loaded truncated SVD factors.  If *None* and `open_svd=True`,
        they are loaded on the fly via
        :pyfunc:`utilities.open_and_reduce_SVD`.
    open_svd : bool, default True
        If *True*, load SVD factors from disk; otherwise use the arrays
        provided via arguments.
    lags : int, default 52
        Length of the lag window used during SHRED training
    forecast : bool, default False
        Placeholder flag (not currently used).
    surface : bool, default False
        If *True*, treat the field as surface elevation rather than
        velocity (plane index forced to zero).
    no_input_u_fluc : bool, default False
        If *True*, `u_fluc` is ignored and re-loaded internally.

    Returns
    -------
    u_fluc_test : ndarray
        Full-rank ground-truth snapshots (`shape = (ny, nx, n_test)`).
    u_svd_test : ndarray
        Corresponding truncated-SVD snapshots 
    u_recons_test : ndarray
        SHRED-reconstructed snapshots 
    u_fluc : ndarray
        The full input cube (returned for convenience, possibly re-loaded).

    Notes
    -----
    * Snapshots are reshaped to `(ny, nx, n_test)` via
      :pyfunc:`convert_2d_to_3d`.
    * For surface fields, set `surface=True` so that plane indices are
      handled correctly.
    * Requires utility functions from `utilities` to locate DNS data and
      compute dimensions.
    """
    experimental_ens=None
    case=None
    shift = test_indices + lags - 1
    num_test_snaps = len(test_indices)
    
    dimX, dimY, dimT = get_dims_DNS(DNS_case)

    #extract test images from original data
    if no_input_u_fluc:
        if surface:
            u_fluc = get_surface_DNS(DNS_case) #calling it u_fluc although it is surf_fluc, due to similar usage in both cases in this function
        else:
            u_fluc = get_velocity_plane_DNS(DNS_case, plane)
            
    u_fluc_test = u_fluc[:,:, shift]

    if open_svd:
        U_tot_red, S_tot_red, V_tot_red = open_and_reduce_SVD(experimental_ens, case, rank, forecast=False, DNS=True, DNS_plane=plane, DNS_surf=surface, DNS_case=DNS_case, exp=False, addr=addr)
    else:
        U_tot_red = U_tot_red[:, plane_index*rank :(plane_index+1)*rank]
        S_tot_red = S_tot_red[plane_index*rank :(plane_index+1)*rank]
        V_tot_red = V_tot_red[:, plane_index*rank :(plane_index+1)*rank]
    
    #construct svd truncation
    u_svd = U_tot_red @ np.diag(S_tot_red) @ np.transpose(V_tot_red)
    u_svd = u_svd[:,shift]
    u_svd_test = convert_2d_to_3d(u_svd, dimY, dimX, num_test_snaps)
    
    #construct reconstructions
    if surface:
        plane=0
        plane_index = 0

    u_recons_test = U_tot_red @ np.diag(S_tot_red) @ np.transpose(V_tot_recons[:, plane_index*rank + num_sensors :(plane_index+1)*rank + num_sensors]) 
    u_recons_test = convert_2d_to_3d(u_recons_test, dimY, dimX, num_test_snaps)

    return u_fluc_test, u_svd_test, u_recons_test, u_fluc


#done
def RMS_plane(data):
    '''calculates RMS amplitude of a 2d plane
    at a single time step'''
    #data is of structure (dimX*dimY, n_test)
    
    sqr = np.power(data, 2)
    data_mean = np.nanmean(sqr, axis=0)
    RMS_data = np.sqrt(data_mean)
    #RMS_data is (4*dimT)
    return RMS_data


#done
def time_avg_RMS(RMS_data):
    '''calculates time average of planar RMS amplitudes time series'''
    #RMS_data is (n_test)
    RMS_time_avg = np.mean(RMS_data,axis=0)
    return RMS_time_avg


#done
def get_RMS(data):
    '''calculates time-averaged RMS values of a 
        3D array 'data' (2D plane with time as 3rd axis)
    '''
    RMS_data = RMS_plane(data)
    RMS_time_avg = time_avg_RMS(RMS_data)
    return RMS_time_avg


#done
def time_avg_RMS_ver2(RMS_data_true, RMS_data_recons):
    '''alternative time-averaging RMS function
    calculates RMS error between plane RMS signals, 
    where error is between RMS of plane per frame, then mean in time'''
    err = RMS_data_true - RMS_data_recons
    RMS_time_avg = np.sqrt(np.mean(np.power(err,2)))
    return RMS_time_avg


'''----------------------------------------------------------------------------------------------------------------------'''


'''GENERAL UTILITY FUNCTIONS FOR OPERATIONS'''

#done
def addr_string(addr_old, addr_new):
    '''changes address type from old to new'''
    if addr_old=='':
        addr = addr_new
    else:
        addr = addr_old
    return addr


#done
def multiply_along_axis(A, B, axis):
    '''to multiply array A with array B along one axis'''
    return np.swapaxes(np.swapaxes(A, axis, -1) * B, -1, axis)

#done
def cross_correlation(a,b):
    '''calculates the normalized cross-correlation
    between array a and b
    '''

    a = (a - np.mean(a)) / (np.std(a) * len(a))
    b = (b - np.mean(b)) / (np.std(b))
    c = np.correlate(a, b, 'full')
    c = c[len(c)//2] #choose the zero lag value
    return c


#done
def convert_3d_to_2d(X):
    """
    Flatten a 3D (Y, X, T) array of 2D fields over time into a 2D (YX, T) matrix.

    Parameters
    ----------
    X : ndarray, shape (Y, X, T)
        3D data with spatial dimensions (Y, X) and time T.

    Returns
    -------
    X_out : ndarray, shape (Y*X, T)
        Columns are time, rows are spatial points flattened in C-order
        (consistent with reshape(..., order='C')).
    """
    time_dim = len(X[0,0])
    Y_dim = len(X)
    X_dim = len(X[0])
    X_out = np.reshape(X, (X_dim*Y_dim, time_dim), order='C')
    return X_out


#done
def convert_2d_to_3d(X, X_dim, Y_dim, time_dim):
    """
    Unflatten a 2D (YX, T) matrix back to a 3D (Y, X, T) array of 2D fields.

    Parameters
    ----------
    X : ndarray, shape (Y_dim*X_dim, time_dim)
        Flattened spatial data with columns as time.
    X_dim : int
        Number of points in the x-direction.
    Y_dim : int
        Number of points in the y-direction.
    time_dim : int
        Number of time snapshots T.

    Returns
    -------
    X_out : ndarray, shape (Y_dim, X_dim, time_dim)
        3D array reconstructed using C-order reshape; inverse of convert_3d_to_2d.
    """
    #X = np.transpose(X)
    X_out = np.reshape(X, (Y_dim, X_dim, time_dim))
    #print("X_out: ", X_out)
    return X_out


#done
def create_GIF(filenames, gif_name):
    '''takes in a list of filename adresses and creates a GIF 
        that is saved with name given by gif_name
    '''
    print("Creating GIF\n")
    with imageio.get_writer(f'{gif_name}.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    print('GIF saved\n')
