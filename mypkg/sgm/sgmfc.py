import numpy as np

class SGMFC():
    def __init__(self, C, D, freqs):
        """args:
            C: connectome matrix
            D: distance matrix
            freqs: the freqs to get the FC, in Hz. 
        """
        if isinstance(freqs, str):
            msg = "only support delta, theta and alpha bands, for other band, plz input a list."
            assert freqs.lower() in ["delta", "theta", "alpha"], msg 
            bands_rgs = {"delta": [2, 3.5], "theta": [4, 7], "alpha": [8, 12]}
            bdlmt = bands_rgs[freqs]
            freqs = np.linspace(bdlmt[0], bdlmt[1], 10)
        self.C = C
        self.D = D
        self.freqs = freqs
        
    def _get_fc_at_freq(self, sgm_params, freq):
        """Network Transfer Function for spectral graph model.
    
        Args:
            sgm_params (dict): sgm_params for ntf. We shall keep this separate from Brain
                   for now, as we want to change and update according to fitting.
            freq (float): frequency at which to calculate NTF, in Hz 
    
        Returns:
            fc(numpy asarray):  The FC for the given frequency (w)
        """
        
        w = 2*np.pi*freq # change from Hz to angular freq
        C = self.C
        D = self.D 
    
        tauC = sgm_params[0]
        speed = sgm_params[1]
        alpha = sgm_params[2]
        
        # Defining some other parameters used:
        zero_thr = 0.01
    
        # define sum of degrees for rows and columns for laplacian normalization
        rowdegree = np.transpose(np.sum(C, axis=1))
        coldegree = np.sum(C, axis=0)
        qind = rowdegree + coldegree < 0.2 * np.mean(rowdegree + coldegree)
        rowdegree[qind] = np.inf
        coldegree[qind] = np.inf
    
        nroi = C.shape[0]
        K = nroi
    
        Tau = 0.001 * D / speed
        Cc = C * np.exp(-1j * Tau * w)
    
        # Eigen Decomposition of Complex Laplacian Here
        L1 = np.identity(nroi)
        L2 = np.divide(1, np.sqrt(np.multiply(rowdegree, coldegree)) + np.spacing(1))
        L = L1 - alpha * np.matmul(np.diag(L2), Cc)
    
        d, v = np.linalg.eig(L)  
        eig_ind = np.argsort(np.abs(d))  # sorting in ascending order and absolute value
        eig_vec = v[:, eig_ind]  # re-indexing eigen vectors according to sorted index
        eig_val = d[eig_ind]  # re-indexing eigen values with same sorted index
    
        eigenvalues = np.transpose(eig_val[:K])
        eigenvectors = eig_vec[:, :K]
    
        # Cortical model
        FG = np.divide(1 / tauC ** 2, (1j * w + 1 / tauC) ** 2)
    
    
        q1 = (1j * w + 1 / tauC * FG * eigenvalues)
        qthr = zero_thr * np.abs(q1[:]).max()
        magq1 = np.maximum(np.abs(q1), qthr)
        angq1 = np.angle(q1)
        q1 = np.multiply(magq1, np.exp(1j * angq1))
        frequency_response = np.divide(1, np.abs(q1)**2)
        
        fc = eigenvectors @ np.diag(frequency_response) @ np.conjugate(eigenvectors.T)
        fc = np.abs(fc)
    
        return fc
    
    def get_fc(self, sgm_params):

        """
        Output:
        estFC, the mean normalized estimated FC at the given frequency computed 
                over the range given in freqrange.
        """
        sgm_params = np.asarray(sgm_params)
        estFC = 0
        for cur_freq in self.freqs:
            cur_estFC = self._get_fc_at_freq(sgm_params, cur_freq)
            estFC = cur_estFC/len(self.freqs) + estFC
    
        # Now normalize estFC
        diagFC = np.diag(np.abs(estFC))
        diagFC = 1./np.sqrt(diagFC)
        D = np.diag( diagFC )
        estFC = np.matmul( D , estFC )
        estFC = np.matmul(estFC , np.matrix.getH(D)) # f_ij/\sqrt(f_ii)\sqrt(f_jj)
        estFC = estFC - np.diag(np.diag( estFC ))
    
        return estFC[:68, :68]

