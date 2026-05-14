import numpy as np

def compute_photogenerated_current(photon_energies, spectral_irradiance, Eg, alpha0, beta0, thickness):
    '''Compute the photogenerated current density from spectral data and absorber parameters.

    Parameters:
      photon_energies : np.ndarray, shape (M,)
          Photon energies in eV, monotonically increasing.
      spectral_irradiance : np.ndarray, shape (M,)
          Spectral irradiance in W/(m^2 eV) at each photon energy.
      Eg : float
          Bandgap energy in eV.
      alpha0 : float
          Absorption constant in cm^{-1}.
      beta0 : float
          Absorption constant in cm^{-1}.
      thickness : float
          Absorber layer thickness in micrometers.

    Returns:
      Jph : float
          Photogenerated current density in A/m^2.
    '''
    #input
    photon_energies = np.asarray(photon_energies, dtype=float)
    spectral_irradiance = np.asarray(spectral_irradiance,dtype=float)

    #mask out 
    spectral_irradiance_mask = np.zeros_like(spectral_irradiance,dtype=float)
    photon_energies_joule = np.zeros_like(photon_energies,dtype=float)
    alpha = np.zeros_like(photon_energies,dtype=float)
    mask = photon_energies > Eg

    #convert eV to joules 
    q = 1.602e-19
    photon_energies_joule[mask] = photon_energies[mask]*q

    #compute alpha
    alpha[mask] = (alpha0 + beta0 * Eg/photon_energies[mask])*np.sqrt(photon_energies[mask]/Eg - 1)
    #compute spectral_irradiance in N photons/s.m2.eV
    spectral_irradiance_mask[mask] = spectral_irradiance[mask]/photon_energies_joule[mask]
    #convert thickness from micrometer to cm
    thickness_cm = thickness*1.0e-4
    
    #compute absorbed photon flux from beer-lambert law along the device thickness
    absorbed_photon_flux = spectral_irradiance_mask * (1.0 - np.exp(-alpha*thickness_cm))
    #compute absorbed photon flux integrating for all energies
    absorbed_photon = np.trapezoid(absorbed_photon_flux,photon_energies)
    #compute the current 
    quantum_efficiency = 1.
    Jph = q*quantum_efficiency*absorbed_photon #Coulomb/s.m2 = A/m2 

    return Jph

def compute_pv_efficiency(photon_energies, spectral_irradiance, Eg, alpha0, beta0, thickness, J0, n_ideality, T, P_in):
    '''Compute PV figures of merit for a perovskite absorber under a given illumination spectrum.

    Parameters:
      photon_energies : np.ndarray, shape (M,)
          Photon energies in eV, monotonically increasing.
      spectral_irradiance : np.ndarray, shape (M,)
          Spectral irradiance in W/(m^2 eV) at each photon energy.
      Eg : float
          Bandgap energy in eV.
      alpha0 : float
          Absorption constant in cm^{-1} (see Eg-sqrt model).
      beta0 : float
          Absorption constant in cm^{-1} (see Eg-sqrt model).
      thickness : float
          Absorber layer thickness in micrometers.
      J0 : float
          Reverse saturation current density in A/m^2.
      n_ideality : float
          Diode ideality factor (dimensionless).
      T : float
          Temperature in Kelvin.
      P_in : float
          Incident power density in W/m^2.

    Returns:
      tuple of (Jsc_mAcm2, Voc, FF, PCE)
          Jsc_mAcm2 : float, short-circuit current density in mA/cm^2
          Voc : float, open-circuit voltage in V
          FF  : float, fill factor (dimensionless)
          PCE : float, power conversion efficiency in percent
    '''
    photon_energies = np.asarray(photon_energies, dtype=float)
    spectral_irradiance = np.asarray(spectral_irradiance,dtype=float)
    Eg = Eg
    alpha0 = alpha0
    beta0 = beta0
    thickness = thickness
    
    q = 1.602e-19
    k_B = 1.381e-23
    
    #compute Jph in A/m^2
    Jph = compute_photogenerated_current(photon_energies, spectral_irradiance, Eg, alpha0, beta0, thickness)
    #ideal one-diode equation model => J = J0 * ( np.exp(q*V/(n_ideality*k_B*T)) - 1 ) - Jph
    #compute Voc using the ideal one-diode equation at J=0 in V
    Voc = ( (n_ideality*k_B*T)/q) * ( np.log(Jph/J0 + 1.0) )
    #compute Jsc_mAcm2 using the ideal one-diode equation at V=0  
    Jsc_mAcm2=0.1*Jph
    #compute maximum power and max power Pm
    V = np.linspace(0, Voc, num=1000000)
    P = -1.0 * ( J0 * ( np.exp( q*V/(n_ideality*k_B*T) ) - 1.0 ) - Jph ) * V
    Pm = np.max(P)
    #compute FF = (Pm/(Jph*Voc))
    den = np.multiply(Jph,Voc)
    if den != 0:
        FF = Pm / den 
    else:
        FF = 0     
    #compute PCE
    PCE = 100*Pm/P_in

    return (Jsc_mAcm2, Voc, FF, PCE)   


if __name__ == "__main__":

    photon_energies = np.array([1.0, 1.2, 1.45, 1.6, 1.85, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0], dtype=float)
    # discrete approximation of the AM1.5G solar spectrum
    # ref: https://compass.astm.org/content-access?contentCode=ASTM%7CG0173-03R20%7Cen-US
    spectral_irradiance = np.array([550, 581, 538, 570, 514, 455, 394, 331, 285, 224, 164], dtype=float) 
    #bandgap NaSnCl3 ref: https://www.sciencedirect.com/science/article/pii/S2352492825017040
    Eg = 1.04
    #absorption coeficients are in same order with ref: https://www.sciencedirect.com/science/article/pii/S2352492824019950
    alpha0 = 3.5e4
    beta0 = 1.2e4
    thickness = 0.5
    J0 = 2.0e-5
    n_ideality = 1.3
    #inputs from ref: https://www.sciencedirect.com/science/article/pii/S2352492825017040
    T = 300.0
    P_in = 1000.0

    print(compute_pv_efficiency(
        photon_energies, spectral_irradiance, Eg, alpha0, beta0,
        thickness, J0, n_ideality, T, P_in
    ))
