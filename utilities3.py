import scipy as sp
import scipy.linalg as la
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import imageio
import math
from PIL import Image, ImageOps
import h5py
import mat73
from lvpyio import read_set
from tkinter import Tk, filedialog
from skimage.filters import window
import scipy.signal as sig
import os



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


def compute_psd(snapshots, dx=1.0, dy=1.0):
    """
    Computes the mean PSD for an array of flow snapshots.

    Parameters:
    - snapshots: 3D NumPy array of shape (num_snapshots, nx, ny) containing flow snapshots.
    - dx: Spacing between grid points in the x-direction.
    - dy: Spacing between grid points in the y-direction.

    Returns:
    - psd_mean: 1D array of the mean PSD as a function of wavenumber.
    - k_mid: 1D array of wavenumber bin midpoints.
    """
    num_snapshots, nx, ny = snapshots.shape

    # Wavenumbers in each direction
    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    k_magnitude = np.sqrt(kx ** 2 + ky ** 2)

    k_max = np.max(k_magnitude)
    k_bins = np.linspace(0, k_max, num=min(nx, ny) // 2)
    k_mid = 0.5 * (k_bins[:-1] + k_bins[1:])

    # Initialize an array for storing the radial PSD
    psd_all = np.zeros((num_snapshots, len(k_mid)))
    #psd_all = []

    #for snap in snapshots:
    for i, snap in enumerate(snapshots):
        # Compute 2D FFT
        fft2 = np.fft.fft2(snap)
        wimage = snap * window('hann', snap.shape)
        fft2 = np.fft.fft2(snap)
        psd2d = np.abs(np.fft.fft2(wimage)) ** 2
        
        #psd2d = np.abs(fft2) ** 2

        # Flatten arrays for binning
        k_flat = k_magnitude.flatten()
        psd_flat = psd2d.flatten()

        # Radial binning
        psd_radial, _ = np.histogram(k_flat, bins=k_bins, weights=psd_flat)
        count, _ = np.histogram(k_flat, bins=k_bins)

        # Avoid division by zero
        psd_radial = psd_radial / (count + 1e-10)
        psd_all[i, :] = psd_radial
        #psd_all.append(psd_radial)

    # Average PSD across all snapshots
    psd_all = np.array(psd_all)
    psd_mean = np.mean(psd_all, axis=0)

    # Compute the midpoints of wavenumber bins for output
    k_mid = 0.5 * (k_bins[:-1] + k_bins[1:])

    return psd_mean, k_mid




def compute_psd_1d(snapshots, dx=1.0, dy=1.0, DNS=False, time_avg=True):
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
        # Compute 2D FFT
        #fft2 = np.fft.fft2(snap)
        #wimage = snap * window('hann', nx)
        #fft2 = np.fft.fft2(snap)
        #psd2d = np.abs(np.fft.fft2(wimage)) ** 2
        
        #data_win=snap
        win = sig.windows.hann(nx,  sym=False)
        data_win = multiply_along_axis(snap, win, axis=0)
        
        data_fft = np.fft.rfft(data_win, n=len(k_mid)*2 -1, axis=0)
        PSD_vals = np.power(abs(data_fft),2)
        #avgeraging spectrum along y (axis 1)
        PSD_vals = np.mean(PSD_vals, axis=1)
        #PSD_vals = np.power(abs(data_fft),2)
        #psd2d = np.abs(fft2) ** 2

        # Flatten arrays for binning
        #k_flat = k_magnitude.flatten()
        #psd_flat = psd2d.flatten()

        # Radial binning
        #psd_radial, _ = np.histogram(k_flat, bins=k_bins, weights=psd_flat)
        #count, _ = np.histogram(k_flat, bins=k_bins)

        # Avoid division by zero
        #psd_radial = psd_radial / (count + 1e-10)
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


def multiply_along_axis(A, B, axis):
    return np.swapaxes(np.swapaxes(A, axis, -1) * B, -1, axis)



def convert_3d_to_2d(X):
    '''function for converting a 3D array of 2d space in time, to 1d snapshots in time'''
    time_dim = len(X[0,0])
    Y_dim = len(X)
    X_dim = len(X[0])
    #print("time_dim: ", time_dim)
    ##print("X_dim: ", X_dim)
    #print("Y_dim: ", Y_dim)

    X_out = np.reshape(X, (X_dim*Y_dim, time_dim), order='C')
    #X_out = np.transpose(X_out)
    #print("out: ", X_out)
    return X_out



def convert_2d_to_3d(X, X_dim, Y_dim, time_dim):
    '''function for converting a 2D array of 1D snapshots in time to 2D space in time'''
    #X = np.transpose(X)
    X_out = np.reshape(X, (Y_dim, X_dim, time_dim))
    #print("X_out: ", X_out)
    return X_out



def create_GIF(filenames, gif_name):
    print("Creating GIF\n")
    with imageio.get_writer(f'{gif_name}.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    print('GIF saved\n')



def write_teetank_to_mat(case='\P25',depth='\H395',num_ensembles=20, dimX=196, dimY=225, dimT = 900,variable='U0',surface=False):
    '''Function to load Teetank data from Davis dataset to MAT-file
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
    tee_fname = "E:\\Users\krissmoe\Documents\PhD data storage\T-Tank\case_"+case+"_"+depth+"_"+variable+".mat" 
    tee_dict = {
        'U': u_plane,
    }

    with h5py.File(tee_fname, 'w') as f:
        for key, value in tee_dict.items():
            f.create_dataset(key, data=value)
    #sp.io.savemat(svd_fname, svd_dict)
    print("DONE!")

def write_teetank_surf_to_mat(surf_data, case, depth):
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

def open_teetank_profilometry(addr):
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
    eta = surf_data.transpose(3, 2, 1, 0)

    return eta


def read_teetank_plane(case='P25',depth='H390',variable='U0',surface=False):

    tee_fname = "E:\\Users\krissmoe\Documents\PhD data storage\T-Tank\case_"+case+"_"+depth+"_"+variable+".mat" 

    with h5py.File(tee_fname, 'r') as tee:
        # List all datasets in the file
        #print("Keys in the HDF5 file:", list(tee.keys()))

        U = np.array(tee['U'])
    return U



def read_teetank_surface(case='P25', depth='H390'):
    surf_fname = "E:\\Users\krissmoe\Documents\PhD data storage\T-Tank\Eta_case_"+case+"_"+depth+".mat" 


    with h5py.File(surf_fname, 'r') as tee:
        # List all datasets in the file
        #print("Keys in the HDF5 file:", list(tee.keys()))

        eta_fluc = np.array(tee['eta'])
    return eta_fluc



def get_surface(DNS_case):
    if DNS_case=='RE2500':
        fname = "E:\\Users\krissmoe\Documents\PhD data storage\VelocityPlanes\\surfElev.mat"
    else:
        fname = "E:\\Users\krissmoe\Documents\PhD data storage\Re1000_WEinf\\surf_elev.mat"
    data = mat73.loadmat(fname)
    if DNS_case=='RE2500':
        eta_full = data['surfElev']
        eta_full = np.transpose(eta_full, (1, 2, 0))
    else:
        eta_full = data['surf_elev']
    eta_mean = np.nanmean(eta_full, axis=2, keepdims=True)

    eta_fluc = eta_full - eta_mean
    return eta_fluc



def get_normalized_surface_DNS(DNS_case):
    eta_fluc = get_surface(DNS_case)
    eta_fluc_2d = convert_3d_to_2d(eta_fluc)
    Xnorm= np.max(np.abs(eta_fluc))
    X = eta_fluc_2d/Xnorm #note, this is surface velocity
    print(X.shape)
    #n2 = (X).shape[0]
    return X

def get_normalized_surface_exp(exp_case, plane, tee_ens):
    eta_fluc = get_surface_teetank(exp_case, plane)
    eta_fluc_2d = convert_3d_to_2d(eta_fluc[:,:,:,tee_ens-1])
    Xnorm= np.max(np.abs(eta_fluc))
    X = eta_fluc_2d/Xnorm #note, this is surface velocity
    print(X.shape)
    #n2 = (X).shape[0]
    return X

def get_velocity_plane_DNS(DNS_case, plane):
    if DNS_case == 'RE2500':
        fname = "E:\\Users\krissmoe\Documents\PhD data storage\VelocityPlanes\\u_layer"+str(plane)+".mat"
        dimX = 256
        dimY=256
        dimT=12500
    else:
        fname = "E:\\Users\krissmoe\Documents\PhD data storage\Re1000_WEinf\\u_layer"+str(plane)+".mat"
        dimX = 128
        dimY=128
        dimT=10900
    data = mat73.loadmat(fname)
    u = data['uPlane']
    u_mean = np.nanmean(u, axis=2, keepdims=True)
    u_fluc = u - u_mean
    return u_fluc

def get_velocity_plane_teetank(teetank_case, plane):
    depths = ['H395', 'H390', 'H375', 'H350', 'H300']
    depth=depths[plane-1] #plane=1 is H395, plane=2 is H390 etc
    u = read_teetank_plane(teetank_case,depth,variable='U0',surface=False)
    #structure of u is (ens, dimY, dimX, dimT)
    u_mean = np.nanmean(u, axis=3, keepdims=True)
    u_fluc = u - u_mean
    return u_fluc

def get_surface_teetank(teetank_case, plane):
    depths = ['H395','H390', 'H375', 'H350', 'H300']
    depth=depths[plane-1] #plane=1 is H395, plane=2 is H390 etc
    eta_fluc = read_teetank_surface(teetank_case, depth)
    return eta_fluc

def get_dims_DNS(DNS_case):
    if DNS_case=='RE2500':
        dimX = 256
        dimY = 256
        dimT = 12500
    else:
        dimX = 128
        dimY = 128
        dimT = 10900
    return dimX, dimY, dimT


def get_dims_teetank_vel():
    dimX = 225
    dimY = 196
    dimT = 900
    return dimX, dimY, dimT

def get_dims_teetank_surf(depth='H390', case='P50'):
    dimX = 610
    dimY = 540
    if depth=='H375' and case=='P50':
        dimY = 543
    dimT = 900
    return dimX, dimY, dimT


def get_mesh_DNS(DNS_case):
    dimX, dimY, dimT = get_dims_DNS(DNS_case)
    X = np.linspace(0,dimX-1, dimX)
    Y = np.linspace(0,dimY-1, dimY) 
    XX, YY = np.meshgrid(X, Y)
    return XX, YY

def get_mesh_Teetank(case='P50', depth='H390'):
    addr = 'E:\\Users\krissmoe\Documents\PhD data storage\T-Tank\surfMesh_' + depth + '_' + case +'.mat'
    meshes = mat73.loadmat(addr)
    #print(meshes)
    X_eta = meshes['xMesh']
    Y_eta = meshes['yMesh']
    #X_vel = meshes['vMesh']['x']
    #Y_vel = meshes['vMesh']['y']
    dim_vel_X = 196
    dim_vel_Y = 225
    X = np.arange(0,dim_vel_X)
    Y = np.arange(0,dim_vel_Y)
    X_vel, Y_vel = np.meshgrid(X,Y)

    return X_eta, Y_eta, X_vel, Y_vel




def get_zz_DNS(DNS_case):
    if DNS_case=='RE2500':

        fname = "E:\\Users\krissmoe\Documents\PhD data storage\VelocityPlanes\\zz.mat"
    else:
        fname = "E:\\Users\krissmoe\Documents\PhD data storage\RE1000_WEinf\\zz_RE1000.mat"
    data = sio.loadmat(fname)
    zz = data['zz'][:,0]

    z = 1 -zz
    print(z)
    z = -z*5*np.pi
    return z

def get_zz_tee():
    z1 = np.array([-0.5, -1.0, -2.5, -5.0, -10.0])
    return z1

def get_integral_length_scale(DNS_case):
    tscales_fname = "E:\\Users\krissmoe\Documents\PhD data storage\SHRED DNS Backup\\TurbScales_" + DNS_case + ".mat"
    TurbScales = sp.io.loadmat(tscales_fname)

    L_int = TurbScales['TurbScales']['Lint'][0,0][0][0]
    L_visc = TurbScales['TurbScales']['Lvisc'][0,0][0][0]
    L_taylor = TurbScales['TurbScales']['Ltay'][0,0][0][0]
    L_kolmogorov = TurbScales['TurbScales']['Lkolm'][0,0][0][0]
    Re_turb = TurbScales['TurbScales']['ReTur'][0,0][0][0]
    Re_taylor = TurbScales['TurbScales']['ReTay'][0,0][0][0]
    u_Rep = TurbScales['TurbScales']['uRep'][0,0][0][0]
    return L_int

def get_normalized_z(z, z_norm, DNS_case):
    '''takes in a z axis (1D array)
        and z_norm argument,
        and normalizes the axis wrt. a length scale
        either None, Taylor length scale, integral scale or mixed

        z_norm: None, 'taylor', 'int', 'mixed'
        '''
    
    #load file with scales:
    tscales_fname = "E:\\Users\krissmoe\Documents\PhD data storage\SHRED DNS Backup\\TurbScales_" + DNS_case + ".mat"
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

def get_normalized_z_tee(z, z_norm, teetank_case):
    '''takes in a z axis (1D array)
        and z_norm argument,
        and normalizes the axis wrt. a length scale
        either None, Taylor length scale, integral scale or mixed

        z_norm: None, 'taylor', 'int', 'mixed'
        '''
    
    #load file with scales:
    #tscales_fname = "E:\\Users\krissmoe\Documents\PhD data storage\SHRED DNS Backup\\TurbScales_" + DNS_case + ".mat"
    #TurbScales = sp.io.loadmat(tscales_fname)
    if teetank_case=='P25':

        L_int = 5.1 #cm
        L_visc = None
        L_taylor = None
    elif teetank_case=='P50':
        L_int = 6.8 #cm
        L_visc = None
        L_taylor = None
    
    if z_norm=='taylor':
        z = z/L_taylor
        print("NO TEETANK TAYLOR BY NOW!")
    elif z_norm=='int':
        z = z/L_int
    elif z_norm=='mixed':
        z = z/np.sqrt(L_int*L_taylor)
    
    return z



def save_svd_full(eta_fluc, u_fluc, ens, case, variable, forecast=False, DNS=False, DNS_plane=None, DNS_surf=False, DNS_case='RE2500', new_teetank=False):
    '''calculates SVD of variable for one ensemble cases, and stack them in U S V'''
    '''assume data_2d is of format (ens, dimx*dimy, dimT)'''


    if DNS:
        #take in one velocity/surface plane and calculates SVD up to rank 1000
        if DNS_surf:
            del u_fluc
            #assume eta_fluc is the 2d version
            U, S, VT_u = np.linalg.svd(eta_fluc[:,:],full_matrices=False)
        else:
            del eta_fluc
            U, S, VT_u = np.linalg.svd(u_fluc[:,:],full_matrices=False)
        #we only store r=1000 modes
        U_tot_u = U[:, :1000]
        del U
        S_tot_u = S[:1000]
        del S
        V_tot = np.transpose(VT_u[:1000,:])
        del VT_u
        #U, S, VT_eta = np.linalg.svd(eta_fluc_2d[:,:],full_matrices=False)
        #U_tot_eta= U[:, :1000]
    else:
        #experimental cases

        planes = ['H300', 'H350', 'H375', 'H390']
            
        for i in range(len(planes)):
            plane_str = planes[i]
            u = read_teetank_plane(case=case,depth=plane_str,variable='U0',surface=False)
            eta_fluc = read_teetank_surface(case=case, depth=plane_str)
            u_fluc = u - np.mean(u, axis=3, keepdims=True)
            del u
            u_fluc = u_fluc[ens-1]
            u_fluc = convert_3d_to_2d(u_fluc)
            eta_fluc = eta_fluc[:,:,:,ens-1]
            eta_fluc = convert_3d_to_2d(eta_fluc)


            #Stack surface elevation on top of one plane at a time!
            #and save this as a separate file!

            U, S, VT = np.linalg.svd(u_fluc,full_matrices=False)
            
            U_tot_u = U
            S_tot_u = S
            V_tot = np.transpose(VT)
        
            U, S, VT = np.linalg.svd(eta_fluc[:,:],full_matrices=False)
            U_tot_eta = U
            S_tot_eta = S
            V_tot = np.hstack((np.transpose(VT),V_tot))
            print("V_tot_dim: ", V_tot.shape)
            
            svd_dict = {
                'U_tot_u': U_tot_u,
                'U_tot_eta': U_tot_eta,
                'S_tot_u': S_tot_u,
                'S_tot_eta': S_tot_eta,
                'V_tot': V_tot}
            adr_loc = "E:\\Users\krissmoe\Documents\PhD data storage\T-Tank"

            svd_fname = adr_loc + "\Teetank SVD_plane_" + plane_str + "_ens"+ str(ens) + "_"+case + ".mat"
            with h5py.File(svd_fname, 'w') as f:
                for key, value in svd_dict.items():
                    f.create_dataset(key, data=value)
            print("saved successfully!")
                
    print("U shape: ", U_tot_u.shape)
    
    adr_loc = "C:\\Users\krissmoe\OneDrive - NTNU\PhD\PhD code\PhD-1\Flow Reconstruction and SHRED\MAT_files"
    
    #save matrices to mat file
    if DNS:
        svd_dict = {
        'U_tot_u': U_tot_u,
        'S_tot_u': S_tot_u,
        'V_tot': V_tot}

        case_str = DNS_case
        if DNS_surf:
            svd_fname = adr_loc + "\SVD_surf_"+case_str+"_WEinf.mat"
        else:
            svd_fname = adr_loc + "\SVD_plane"+ str(DNS_plane)+ "_"+case_str+"_WEinf.mat"
    elif not new_teetank:
        svd_dict = {
        'U_tot_u': U_tot_u,
        'U_tot_eta': U_tot_eta,
        'S_tot_u': S_tot_u,
        'S_tot_eta': S_tot_eta,
        'V_tot': V_tot}
        adr_loc = "C:\\Users\krissmoe\OneDrive - NTNU\PhD\PhD code\PhD-1\Flow Reconstruction and SHRED\MAT_files"
    
    else:
        svd_fname = adr_loc + "\Teetank SVD_fullplanes_" + variable + "_ens"+ str(ens) + "_"+case + ".mat"
    with h5py.File(svd_fname, 'w') as f:
        for key, value in svd_dict.items():
            f.create_dataset(key, data=value)
    #sp.io.savemat(svd_fname, svd_dict)
    print("DONE!")
    if DNS:
        return U_tot_u, S_tot_u, V_tot
    else:
        return U_tot_u, U_tot_eta, S_tot_u, S_tot_eta, V_tot


def save_singular_values_full(DNS_case, DNS_plane):
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
    adr_loc = "C:\\Users\krissmoe\OneDrive - NTNU\PhD\PhD code\PhD-1\Flow Reconstruction and SHRED\MAT_files"
    S_fname = adr_loc + "S_fullrank_"+ DNS_case + "_plane"+str(DNS_plane)
    with h5py.File(S_fname, 'w') as f:
        for key, value in S_dict.items():
            f.create_dataset(key, data=value)

def save_singular_values_full_tee(teetank_case, ensemble, plane):
    u_fluc = get_velocity_plane_teetank(teetank_case, plane)
    u_fluc = u_fluc[ensemble-1]
    print("Starting SVD")
    u_fluc=convert_3d_to_2d(u_fluc)
    U, S, VT = np.linalg.svd(u_fluc,full_matrices=False)
    print(S.shape)
    print("SVD finished")
    del U
    del VT
    S_dict = {
                    'S': S}
    adr_loc = "C:\\Users\krissmoe\OneDrive - NTNU\PhD\PhD code\PhD-1\Flow Reconstruction and SHRED\MAT_files"
    S_fname = adr_loc + "S_fullrank_teetank_"+ teetank_case + "_ens" + str(ensemble) +  "_plane"+str(plane)
    with h5py.File(S_fname, 'w') as f:
        for key, value in S_dict.items():
            f.create_dataset(key, data=value)

def get_singular_values_full(DNS_case, DNS_plane):
    adr_loc = "C:\\Users\krissmoe\OneDrive - NTNU\PhD\PhD code\PhD-1\Flow Reconstruction and SHRED\MAT_files"
    S_fname = adr_loc + "S_fullrank_"+ DNS_case + "_plane"+str(DNS_plane)
    with h5py.File(S_fname, 'r') as s_matrix:
        # List all datasets in the file
        S = np.array(s_matrix['S'])

    return S

def get_singular_values_full_tee(teetank_case, ensemble, plane):
    adr_loc = "C:\\Users\krissmoe\OneDrive - NTNU\PhD\PhD code\PhD-1\Flow Reconstruction and SHRED\MAT_files"
    S_fname = adr_loc + "S_fullrank_teetank_"+ teetank_case + "_ens" + str(ensemble) +  "_plane"+str(plane)
    with h5py.File(S_fname, 'r') as s_matrix:
        # List all datasets in the file
        S = np.array(s_matrix['S'])

    return S


def get_cumsum_svd(r_vals, total_ranks, DNS_case):
    s_fname = "C:\\Users\krissmoe\OneDrive - NTNU\PhD\PhD code\PhD-1\Flow Reconstruction and SHRED\MAT_files\S_" + DNS_case +".mat" 
    with h5py.File(s_fname, 'r') as s_matrix:
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

def get_cumsum_svd_tee(r_vals, total_ranks, teetank_case, ensemble, plane):
    adr_loc = "C:\\Users\krissmoe\OneDrive - NTNU\PhD\PhD code\PhD-1\Flow Reconstruction and SHRED\MAT_files"
    S_fname = adr_loc + "S_fullrank_teetank_"+ teetank_case + "_ens" + str(ensemble) +  "_plane"+str(plane)
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



def calculate_DNS_SVDs(plane_start, plane_end, DNS_case='RE2500'):
    '''Function that calculates SVD matrices for a given
        set of DNS planes, and saves these in mat files'''
    num_planes=plane_end-plane_start
    eta_fluc_2d = np.zeros(1)#utilities.convert_3d_to_2d(u_fluc)
    ens=0
    variable='u'
    case=None
    for plane in range(plane_start, plane_end+1):
        print("plane: ", plane)
        if DNS_case == 'RE2500':
            fname = "E:\\Users\krissmoe\Documents\PhD data storage\VelocityPlanes\\u_layer"+str(plane)+".mat"
        else:
            print("test")
            fname = "E:\\Users\krissmoe\Documents\PhD data storage\Re1000_WEinf\\u_layer"+str(plane)+".mat"
        data = mat73.loadmat(fname)
        u = data['uPlane']
        print(u.shape)
        del data
        u_mean = np.nanmean(u, axis=2, keepdims=True)
        u_fluc = u - u_mean

        del u
        del u_mean
        

        u_fluc = convert_3d_to_2d(u_fluc)
        print("start svd")
        U, S, V = save_svd_full(eta_fluc_2d, u_fluc, ens, case, variable, forecast=False, DNS=True, DNS_plane=plane, DNS_case=DNS_case, DNS_surf=False)
        print("U: ", U.shape)
        print("S: ", S.shape)
        print("V: ", V.shape)
        del U, S, V
    


def open_SVD(r, ens, vel_fluc=False, variable='u', Teetank=False, teetank_case=None, forecast=False, DNS_new=False, DNS_plane=None, DNS_surf=False, DNS_case='RE2500', Tee_plane='H390'):
    '''Opens and loads file containing SVD matrices for a given velocity plane or surface elevation'''
    
    adr_loc = "C:\\Users\krissmoe\OneDrive - NTNU\PhD\PhD code\PhD-1\Flow Reconstruction and SHRED\MAT_files"
    if Teetank:
        adr_loc = "E:\\Users\krissmoe\Documents\PhD data storage\T-Tank"
                
        svd_fname = adr_loc + "\Teetank SVD_plane_" + Tee_plane + "_ens"+ str(ens) + "_"+teetank_case + ".mat"
        if forecast:
            print("finds filename")
            svd_fname = adr_loc + "\Teetank_Forecast_SVD_fullplanes_" + variable + "_"+teetank_case + ".mat"
    elif DNS_new:
        if DNS_surf:
            svd_fname = adr_loc + "\SVD_surf_"+DNS_case+"_WEinf.mat"
        else:
            svd_fname = adr_loc + "\SVD_plane"+ str(DNS_plane) +"_"+DNS_case+"_WEinf.mat"
    else:
        svd_fname = adr_loc + "\SVD_r"+ str(r) +"_"+DNS_case+"_WEinf.mat"

    #load U, S, V matrices
    with h5py.File(svd_fname, 'r') as SVD:
        U_tot_u = np.array(SVD['U_tot_u']) #U matrix
        S_tot_u = np.array(SVD['S_tot_u'])
        
        if Teetank:
            #for experimental data, U and S matrices are separated into
            #one set for velocity field (ending with _u)
            #another set for surface elevation (ending with _eta)

            U_tot_eta = np.array(SVD['U_tot_eta']) 
            S_tot_eta = np.array(SVD['S_tot_eta'])
        V_tot = np.array(SVD['V_tot'])
        if vel_fluc:
            u_fluc = np.array(SVD['u_fluc'])
            return U_tot_u, S_tot_u, U_tot_eta, S_tot_eta, V_tot, u_fluc
        else:
            if Teetank:
                
                return U_tot_u, S_tot_u, U_tot_eta, S_tot_eta, V_tot
            else:
                return U_tot_u, S_tot_u, V_tot



def reduce_SVD(U, S, V, levels, r_new, Tee=True, DNS_new=False, surf=False):
    '''Function to truncate a high-rank SVD to a lower rank SVD, as given by new rank value r_new'''
    if DNS_new:
        levels=1 
    r = V.shape[1]//levels
    
    if not Tee:
        U_tot_new = np.zeros((U.shape[0],r_new*levels))
        S_tot_new = np.zeros(r_new*levels)
    else:
        if surf==True:
            U_tot_new = np.zeros((U.shape[0],r_new))
            S_tot_new = np.zeros(r_new)
        else:
            
            U_tot_new = np.zeros((U.shape[0],r_new))
            #U_tot_new = np.zeros((U.shape[0],r_new*(levels-1)))
            S_tot_new = np.zeros(r_new)
    V_tot_new = np.zeros((V.shape[0],r_new*levels))
    for i in range(levels):
        if not Tee:
            U_tot_new[:, i*r_new:(i+1)*r_new] = U[:, i*r:i*r+r_new]
            S_tot_new[i*r_new:(i+1)*r_new] = S[i*r:i*r + r_new]
        if surf==False:
            #if i<levels-1:
            #    U_tot_new[:, i*r_new:(i+1)*r_new] = U[:, i*r:i*r+r_new]
            #    S_tot_new[i*r_new:(i+1)*r_new] = S[i*r:i*r + r_new]
            U_tot_new[:, 0:r_new] = U[:, 0:r_new]
            S_tot_new[0:r_new] = S[0:r_new]
        V_tot_new[:, i*r_new:(i+1)*r_new] = V[:, i*r:i*r + r_new]

    if Tee:
        if surf==True:
            U_tot_new[:,:r_new] = U[:, :r_new]
            S_tot_new[0:r_new] = S[0:r_new]

    return U_tot_new, S_tot_new, V_tot_new



def open_and_reduce_SVD(teetank_ens, teetank_case, r, r_new, forecast=False, DNS_new=False, DNS_plane=None, DNS_surf=False, DNS_case='RE2500', Teetank=True, Tee_plane='H390'):
    '''Opens files containing SVD matrices for a given velocity plane or surface elevation, 
        and truncates the U,S,V-matrices with the specified rank truncation r_new'''
    if DNS_new:
        U_tot_u, S_tot_u, V_tot = open_SVD(r, teetank_ens, False, 'u', Teetank, teetank_case, forecast, DNS_new, DNS_plane, DNS_surf, DNS_case=DNS_case)
        levels=1
    else:
        
        U_tot_u, S_tot_u, U_tot_eta, S_tot_eta, V_tot = open_SVD(r, teetank_ens, False, 'u', Teetank,  teetank_case, forecast, DNS_new, DNS_plane, DNS_surf, Tee_plane=Tee_plane)
        levels=2
        #extract reduced surface field
        U_tot_eta_red, S_tot_eta_red, V_tot_red = reduce_SVD(U_tot_eta, S_tot_eta, V_tot, levels, r_new, Teetank, DNS_new, surf=True)
        
    U_tot_u_red, S_tot_u_red, V_tot_red = reduce_SVD(U_tot_u, S_tot_u, V_tot, levels, r_new, Teetank, DNS_new, surf=False)
    
    if DNS_new:
        return U_tot_u_red, S_tot_u_red, V_tot_red
    else:
        
        return U_tot_u_red, S_tot_u_red, U_tot_eta_red, S_tot_eta_red, V_tot_red



def open_SHRED(teetank_ens, case, r, num_sensors, SHRED_ens, plane_list, DNS=True, Tee_plane='H390', full_planes=True, forecast=False, DNS_case='RE2500'):
    '''loads V matrix for test data from SHRED runs, specified by rank value r, number of sensors (num_sensors) and the SHRED ensemble-case SHRED_ens
        returns 
        test_recons: V matrix for reconstruction 
        test_ground_truth: V Matrix for rank r-compressed test data
        test_indices: the indices in the full dataset where test data is extracted from'''
    
    adr_loc = "C:\\Users\krissmoe\OneDrive - NTNU\PhD\PhD code\PhD-1\Flow Reconstruction and SHRED\MAT_files"
    
    if DNS:
        if not full_planes:
            plane_string = "_planes"
            for i in range(len(plane_list)):
                plane_string = plane_string + "_" +  str(plane_list[i]) 
        else:
            plane_string ="_full_planes"
        if forecast==False:
            SHRED_fname = adr_loc + "\SHRED_r"+ str(r) +"_" +str(num_sensors) +"sensors_ens" + str(SHRED_ens) + plane_string +".mat"
            if DNS_case=='RE1000':
                SHRED_fname = adr_loc + "\SHRED_RE1000_r"+ str(r) +"_" +str(num_sensors) +"sensors_ens" + str(SHRED_ens) + plane_string +".mat"
        else:
            SHRED_fname = adr_loc + "\SHRED_FORECAST_r"+ str(r) +"_" +str(num_sensors) +"sensors_ens" + str(SHRED_ens) + plane_string +".mat"
            if DNS_case=='RE1000':
                SHRED_fname = adr_loc + "\SHRED_FORECAST_RE1000_r"+ str(r) +"_" +str(num_sensors) +"sensors_ens" + str(p) + plane_string + ".mat"
                
    else:
        if forecast:
            print("must change SHRED filename for Teetank forecast")
            SHRED_fname = adr_loc + "\Teetank_FORECAST_case_"+ case + "_r" + str(r) +"_" +str(num_sensors) +"sensors_ens" + str(SHRED_ens) + ".mat"
        else:
            SHRED_fname = adr_loc + "\Teetank_SHRED_new_ens"+ str(teetank_ens) + "_"+ case + "_" + Tee_plane + "_r"+ str(r) +"_" +str(num_sensors) +"sensors_ens" + str(SHRED_ens) +".mat"
    with h5py.File(SHRED_fname, 'r') as SHRED:
        # List all datasets in the file
        #print("Keys in the HDF5 file:", list(SHRED.keys()))

        test_recons = np.array(SHRED['test_recons'])
        test_ground_truth = np.array(SHRED['test_ground_truth'])
        test_indices = np.array(SHRED['test_indices'])

    return test_recons, test_ground_truth, test_indices



def get_test_imgs_SHRED_Teetank1(eta_fluc, u_fluc, teetank_ens, teetank_case,r, r_new, SHRED_ens, num_sensor, X_vel, X_eta, dimT, num_ensembles, lags=52, forecast=False):
            
    V_tot_recons, V_tot_svd, test_indices2 = open_SHRED(teetank_ens, teetank_case, r_new, num_sensor, SHRED_ens, forecast)

    shift = test_indices2 + lags - 1

    if forecast:
        #shift indices
        shift = shift - dimT*(num_ensembles-1)

    #print("shift1: ", shift)
    num_test_snaps = len(test_indices2)
            
    
    
    eta_fluc_test = eta_fluc[teetank_ens-1, :,:,:]
    eta_fluc_test = eta_fluc_test[:, :, shift]
    u_fluc_test = u_fluc[teetank_ens-1, :, :,:]
    del u_fluc
    del eta_fluc
    
    u_fluc_test = u_fluc_test[:,:, shift]
    

    if forecast:
        U_u, S_u, U_eta, S_eta, V_tot_red= open_and_reduce_SVD(teetank_ens, teetank_case, 1000, r_new, forecast=True)
    
        #Extract SVD fields
        eta_svd = U_eta@ np.diag(S_eta) @ np.transpose(V_tot_svd[:, 0+num_sensor:r_new+num_sensor])
        #eta_svd = eta_svd[:,test_indices2 + lags - 1]
        print("eta_svd_shape0: ", eta_svd.shape)
        eta_svd_test = convert_2d_to_3d(eta_svd, X_eta.shape[1], X_eta.shape[0], num_test_snaps)
        del eta_svd
        print("eta_svd_shape: ", eta_svd_test.shape)
        u_svd = U_u @ np.diag(S_u) @ np.transpose(V_tot_svd[:, r_new + num_sensor:2*r_new+num_sensor])
        #u_svd = u_svd[:,test_indices2 + lags - 1]
        u_svd_test = convert_2d_to_3d(u_svd, X_vel.shape[1], X_vel.shape[0], num_test_snaps)

        #construct reconstruction
        u_recons_test = U_u @ np.diag(S_u) @ np.transpose(V_tot_recons[:, r_new + num_sensor :2*r_new + num_sensor])
        del U_u, S_u
        u_recons_test = convert_2d_to_3d(u_recons_test, X_vel.shape[1], X_vel.shape[0], num_test_snaps)

        
        eta_recons_test = U_eta @ np.diag(S_eta) @ np.transpose(V_tot_recons[:, 0 + num_sensor :r_new + num_sensor])
        del U_eta, S_eta
        eta_recons_test = convert_2d_to_3d(eta_recons_test, X_eta.shape[1], X_eta.shape[0], num_test_snaps)
    else:
        #Extract SVD fields
        
        U_tot_u_red, S_tot_u_red, U_tot_eta_red, S_tot_eta_red, V_tot_red = open_and_reduce_SVD(teetank_ens, teetank_case, r, r_new)
        eta_svd = U_tot_eta_red @ np.diag(S_tot_eta_red) @ np.transpose(V_tot_red[:, 0:r_new])
        eta_svd = eta_svd[:,shift]
        eta_svd_test = convert_2d_to_3d(eta_svd, X_eta.shape[1], X_eta.shape[0], num_test_snaps)
        del eta_svd
        #print("eta_svd_shape: ", eta_svd_test.shape)
        u_svd = U_tot_u_red @ np.diag(S_tot_u_red) @ np.transpose(V_tot_red[:, r_new:2*r_new])
        u_svd = u_svd[:,shift]
        u_svd_test = convert_2d_to_3d(u_svd, X_vel.shape[1], X_vel.shape[0], num_test_snaps)
    
        #print("u_svd: ", u_svd.shape)
        del u_svd

        #construct reconstructions
        eta_recons_test = U_tot_eta_red @ np.diag(S_tot_eta_red) @np.transpose(V_tot_recons[:, 0 + num_sensor :r_new + num_sensor])
        del U_tot_eta_red, S_tot_eta_red
        eta_recons_test = convert_2d_to_3d(eta_recons_test, X_eta.shape[1], X_eta.shape[0], num_test_snaps)

        u_recons_test = U_tot_u_red @ np.diag(S_tot_u_red) @ np.transpose(V_tot_recons[:, r_new + num_sensor :2*r_new + num_sensor])
        del U_tot_u_red, S_tot_u_red, V_tot_recons
        u_recons_test = convert_2d_to_3d(u_recons_test, X_vel.shape[1], X_vel.shape[0], num_test_snaps)

    return eta_fluc_test, u_fluc_test, eta_recons_test, u_recons_test, eta_svd_test, u_svd_test



def get_test_imgs_SHRED_Teetank(Tee_plane, eta_fluc, u_fluc, V_tot_recons, V_tot_svd, test_indices, X_eta, X_vel, teetank_ens, teetank_case,r, r_new, SHRED_ens, num_sensor, U_tot_red=None, S_tot_red=None, V_tot_red = None, open_svd=True, lags=52, forecast=False, surface=False,no_input_u_fluc=False):
            
    #V_tot_recons, V_tot_svd, test_indices2 = open_SHRED(teetank_ens, teetank_case, r_new, num_sensor, SHRED_ens, forecast)

    shift = test_indices + lags - 1

    if forecast:
        dimT=900

        #shift indices
        shift = shift - dimT*(num_ensembles-1)

    print("shift1: ", shift)
    num_test_snaps = len(test_indices)
            
    
    if no_input_u_fluc:
        if surface:
            print("load surface")
            eta_fluc = read_teetank_surface(case=teetank_case, depth=Tee_plane)
            u_fluc = eta_fluc[:,:,:,teetank_ens-1]
        else:
            print("load velocity field, plane " + Tee_plane)
            
            u =read_teetank_plane(case=teetank_case,depth=Tee_plane,variable='U0',surface=False)
            
            u_mean = np.nanmean(u, axis=3, keepdims=True)
            u_fluc = u - u_mean
            u_fluc = u_fluc[teetank_ens-1]
            


    
    u_fluc_test = u_fluc[:,:, shift]
    

    if forecast:
        #TODO fix forecast for teetank

        U_u, S_u, U_eta, S_eta, V_tot_red= open_and_reduce_SVD(teetank_ens, teetank_case, 1000, r_new, forecast=True)
    
        #Extract SVD fields
        eta_svd = U_eta@ np.diag(S_eta) @ np.transpose(V_tot_svd[:, 0+num_sensor:r_new+num_sensor])
        #eta_svd = eta_svd[:,test_indices2 + lags - 1]
        print("eta_svd_shape0: ", eta_svd.shape)
        eta_svd_test = convert_2d_to_3d(eta_svd, X_eta.shape[1], X_eta.shape[0], num_test_snaps)
        del eta_svd
        print("eta_svd_shape: ", eta_svd_test.shape)
        u_svd = U_u @ np.diag(S_u) @ np.transpose(V_tot_svd[:, r_new + num_sensor:2*r_new+num_sensor])
        #u_svd = u_svd[:,test_indices2 + lags - 1]
        u_svd_test = convert_2d_to_3d(u_svd, X_vel.shape[1], X_vel.shape[0], num_test_snaps)

        #construct reconstruction
        u_recons_test = U_u @ np.diag(S_u) @ np.transpose(V_tot_recons[:, r_new + num_sensor :2*r_new + num_sensor])
        del U_u, S_u
        u_recons_test = convert_2d_to_3d(u_recons_test, X_vel.shape[1], X_vel.shape[0], num_test_snaps)

        
        eta_recons_test = U_eta @ np.diag(S_eta) @ np.transpose(V_tot_recons[:, 0 + num_sensor :r_new + num_sensor])
        del U_eta, S_eta
        eta_recons_test = convert_2d_to_3d(eta_recons_test, X_eta.shape[1], X_eta.shape[0], num_test_snaps)
    else:
        #Extract SVD fields
        if open_svd:
            U_tot_u_red, S_tot_u_red, U_tot_eta_red, S_tot_eta_red, V_tot_red = open_and_reduce_SVD(teetank_ens, teetank_case, r, r_new, forecast=False, DNS_new=False, DNS_plane=None, DNS_surf=False, Teetank=True, Tee_plane=Tee_plane)

            if surface:
                U_tot_red = U_tot_eta_red
                S_tot_red = S_tot_eta_red
            else:
                U_tot_red = U_tot_u_red
                S_tot_red = S_tot_u_red
        
        if surface:
            plane_index=0
            dimY = X_eta.shape[1]
            dimX = X_eta.shape[0]
        else:
            plane_index=1
            dimY = X_vel.shape[1]
            dimX = X_vel.shape[0]
        
        V_tot_red = V_tot_red[:, plane_index*r_new :(plane_index+1)*r_new]
        V_tot_recons = V_tot_recons[:, plane_index*r_new + num_sensor :(plane_index+1)*r_new + num_sensor]
        
        #construct svd truncation
        u_svd = U_tot_red @ np.diag(S_tot_red) @ np.transpose(V_tot_red)
        u_svd = u_svd[:,shift]
        u_svd_test = convert_2d_to_3d(u_svd, dimY, dimX, num_test_snaps)

         #construct reconstructions
        u_recons_test = U_tot_red @ np.diag(S_tot_red) @ np.transpose(V_tot_recons) 
        u_recons_test = convert_2d_to_3d(u_recons_test, dimY, dimX, num_test_snaps)

    return u_fluc_test, u_svd_test, u_recons_test, u_fluc



def get_test_imgs_SHRED_DNS(DNS_case, plane, plane_index, u_fluc, V_tot_recons, test_indices, r, r_new, num_sensors, U_tot_red=None, S_tot_red=None, V_tot_red = None, open_svd=True, lags=52, forecast=False, surface=False, no_input_u_fluc=False):
    '''takes in V matrix from SHRED, as well as raw data, and gives velocity fields out'''
    teetank_ens=None
    case=None
    shift = test_indices + lags - 1
    num_test_snaps = len(test_indices)
    
    dimX, dimY, dimT = get_dims_DNS(DNS_case)

    #extract test images from original data
    if no_input_u_fluc:
        if surface:
            u_fluc = get_surface(DNS_case) #calling it u_fluc although it is eta_fluc, due to similar usage in both cases in this function
        else:
            u_fluc = get_velocity_plane_DNS(DNS_case, plane)
            
    u_fluc_test = u_fluc[:,:, shift]

    if open_svd:
        U_tot_red, S_tot_red, V_tot_red = open_and_reduce_SVD(teetank_ens, case, r, r_new, forecast=False, DNS_new=True, DNS_plane=plane, DNS_surf=surface, DNS_case=DNS_case, Teetank=False)
    else:
        U_tot_red = U_tot_red[:, plane_index*r_new :(plane_index+1)*r_new]
        S_tot_red = S_tot_red[plane_index*r_new :(plane_index+1)*r_new]
        V_tot_red = V_tot_red[:, plane_index*r_new :(plane_index+1)*r_new]
    
    #construct svd truncation
    u_svd = U_tot_red @ np.diag(S_tot_red) @ np.transpose(V_tot_red)
    u_svd = u_svd[:,shift]
    u_svd_test = convert_2d_to_3d(u_svd, dimY, dimX, num_test_snaps)
    
    #construct reconstructions
    if surface:
        plane=0
        plane_index = 0
    
    print("U_shape: ", U_tot_red.shape)
    print("S shape: ", S_tot_red.shape)
    print("V shape: ", V_tot_recons.shape)
    u_recons_test = U_tot_red @ np.diag(S_tot_red) @ np.transpose(V_tot_recons[:, plane_index*r_new + num_sensors :(plane_index+1)*r_new + num_sensors]) 
    u_recons_test = convert_2d_to_3d(u_recons_test, dimY, dimX, num_test_snaps)

    return u_fluc_test, u_svd_test, u_recons_test, u_fluc



def RMS_plane(data):
    '''calculates RMS amplitude of a 2d plane
    at a single time step'''
    #data is of structure (dimX*dimY, n_test)
    
    sqr = np.power(data, 2)
    data_mean = np.nanmean(sqr, axis=0)
    RMS_data = np.sqrt(data_mean)
    #RMS_data is (4*dimT)
    return RMS_data



def time_avg_RMS(RMS_data):
    '''calculates time average of planar RMS amplitudes time series'''
    #RMS_data is (n_test)
    RMS_time_avg = np.mean(RMS_data,axis=0)
    return RMS_time_avg



def get_RMS(data):
    RMS_data = RMS_plane(data)
    RMS_time_avg = time_avg_RMS(RMS_data)
    return RMS_time_avg



def time_avg_RMS_ver2(RMS_data_true, RMS_data_recons):
    '''alternative time-averaging RMS function
    calculates RMS error between plane RMS signals, 
    where error is between RMS of plane per frame, then mean in time'''
    err = RMS_data_true - RMS_data_recons
    RMS_time_avg = np.sqrt(np.mean(np.power(err,2)))
    return RMS_time_avg