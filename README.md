# General Design of the SpecAI Package

SpecAI is a pacakge for spectroscopy simulation and chemical applications backed up by the AI technology. It is planned to cover almost all established spectroscopy techniques with broad ranges of chemical applications, which include but not limit to
- Vibrational spectroscopy
  - Infrared (IR) absorption
  - Optical Raman spectroscopy
- Valence electron spectroscopy
  - Ultraviolet-visible (UV-vis) absorption and emission
- Core electron spectroscopy
  - X-ray absorption (K-edge, L-edge, etc)
  - X-ray emission
  - Resonant Inelastic X-ray Scattering (RIXS)
- Magnetic Resonance Spectroscopy
  - Nuclear Magnetic Resonance (NMR)
  - Electron paramagnetic resonance (EPR) or electron spin resonance (ESR)
- Chiroptical Spectroscopy
  - Circular dichroism (CD)
    - Vibrational circular dichroism (VCD)
    - Electronical circular dichroism (ECD)
    - X-ray natural circular dichroism (XNCD)
    - X-ray magnetic circular dichroism (XMCD)
  - Optical activity (OA)
    - Optical Raman optical activity (ROA)
    - Vibrational optical activity (VOA)
- Multidimensional and nonlinear spesctroscopy
  - Multidimentional spectroscopy
    - Two-dimensional Infrared (2DIR) spectroscopy
    - Two-dimensional ultraviolet (2DUV) spectroscopy
  - Nonlinear and ultrafast spectroscopy
    - Two-photon absorption (TPA)
    - Four-wave mixing
    - Time-resolved X-ray absoprtion (TRXAS) or X-ray transient absorption spectroscopy (XTAS)

Some of the common spectroscopy signals, such as the IR and UV-vis absorption, can be simulated routinely using standard quantum chemistry packages. However, some signals are poorly supported by quantum chemistry packages (e. g., RIXS), or not supported at all (e. g., four-wave mixing). SpecAI aims at building a bridge between quantum chemistry theory and spectroscopy application. The users are not necessarily to be experts in both quantum chemistry and spectroscopy, but still can find the proper theoretical tool, correctly simulate the signals, nicely present the simulation results, and painlessly interpret the spectroscopy features. In addtion, high throughput calcualtion workflows will be implemented in SpecAI, which faciliate to generate big data of theoretical spectroscopy. Together with experiment data and popular machine learning (ML) packages such as PyTorch or TensorFlow, SpecAI can be used to design spectroscopic descriptors and build ML models to predict molecular properties from inexpensive simulations.

Wheels will not be re-invented in SpecAI. SpecAI calls an external quantum chemistry program (QCEngine) to determine the spectroscopy signal if they are avaialbe in the QCEngine. Multiple popular open-source or free quantum chemistry packages are in the radar of SpecAI, such as ORCA, Psi4 and NWChem. If the target spectroscopy signal is not available in any supported QCEngine, SpecAI would try to obtain the necessary quantities in order to simulate the signal, for example, molecular orbitals, quantum chemistry integrals, excited state energies, etc. With those necessary quantities, SpecAI would run the spectroscopy simulation with its modules. In this way, SpecAI is expected to cover as many spectroscopy signals as possible.

SpecAI is a Python package which can call QCEngines and postprocess their results. QCEngine calcualtions should be submitted to a queue system on some HPC system. Calculation results should be retrieved and data should be extracted and stored in formats facilitating future use in machine learning modeling. The simulated spectroscopy signals can be transformed and plotted with the Python packages Numpy and Matplotlib.
