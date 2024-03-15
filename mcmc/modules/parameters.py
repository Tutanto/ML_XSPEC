import numpy as np

class Parameters:
    def __init__(self, nH=1, Betor10=-2, Rin_M=10, Incl=30, rel_refl=-1, Fe_abund=1, log_xi=1, kTs=2, alpha=2, kTe=2, norm=1, Tin=1, norm_disk=np.log10(1), f_true=0.2):
        # Parameters
        self.nH = nH
        self.Betor10 = Betor10
        self.Rin_M = Rin_M
        self.Incl = Incl
        self.rel_refl = rel_refl
        self.Fe_abund = Fe_abund
        self.log_xi = log_xi
        self.kTs = kTs
        self.alpha = alpha
        self.kTe = kTe
        self.norm = norm
        self.Tin = Tin
        self.norm_disk = norm_disk
        self.f_true = f_true
        
        # Ranges for each parameter (min, max)
        self.ranges = {
            'nH': (0.1, 10),
            'Betor10': (-3, -1),
            'Rin_M': (6, 150),
            'Incl': (0, 90),
            'rel_refl': (-1, 0),
            'Fe_abund': (0.5, 3.0),
            'log_xi': (1, 4),
            'kTs': (0.1, 1.1),
            'alpha': (0.1, 1.5),
            'kTe': (1.1, 20),
            'norm': (0.1, 10),
            'Tin': (0.1, 1.1),
            'norm_disk': (-1, 6),
            'f_true': (-5, 1)
        }

    def to_array(self):
        """ Convert parameters to an array. """
        return np.array([self.nH, self.Betor10, self.Rin_M, self.Incl, self.rel_refl, self.Fe_abund, self.log_xi, self.kTs, self.alpha, self.kTe, self.norm, self.Tin, self.norm_disk, self.f_true])

    def is_param_within_range(self, param_name):
        """ Check if a specific parameter is within its defined range. """
        if param_name in self.ranges:
            value = getattr(self, param_name)
            min_val, max_val = self.ranges[param_name]
            return min_val <= value <= max_val
        return True  # If the parameter is not in ranges, assume no range restriction
    
    def update_from_array(self, array):
        """ Update parameters from an array (e.g., the output of an optimization). """
        (self.nH, self.Betor10, self.Rin_M, self.Incl, self.rel_refl, 
         self.Fe_abund, self.log_xi, self.kTs, self.alpha, self.kTe, 
         self.norm, self.Tin, self.norm_disk, self.f_true) = array

    def __repr__(self):
        """ String representation for easy debugging, one parameter per line. """
        params_repr = '\n'.join([f'{key}={value}' for key, value in self.__dict__.items() if key != 'ranges'])
        return f'Parameters(\n{params_repr}\n)'