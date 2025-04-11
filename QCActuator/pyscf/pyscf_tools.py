import yaml
import numpy as np
from pyscf import gto, scf, tdscf


class PySCFCalculator:
    def __init__(self, config_file=None):
        """Initialize the PySCF calculator with optional configuration file."""
        self.config = {}
        if config_file:
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)

    def run_spectroscopy(self, spec_type, geometry):
        """Run spectroscopy calculation based on the specified type.

        Args:
            spec_type (str): Type of spectroscopy (e.g., "UV")
            geometry (str or dict): Molecular geometry in XYZ format or as a dictionary

        Returns:
            dict: Spectroscopic data including excitation energies and oscillator strengths
        """
        if spec_type != "UV":
            raise ValueError(
                f"Spectroscopy type {spec_type} not supported. Only UV is currently implemented.")

        # Parse configuration
        basis = self.config.get('basis', '6-31g')
        xc_functional = self.config.get('xc_functional', 'b3lyp')
        n_states = self.config.get('n_states', 10)

        # Build molecule
        mol = self._build_molecule(geometry)

        # Run SCF calculation
        mf = scf.RKS(mol)
        mf.xc = xc_functional
        mf.kernel()

        # Run TDDFT calculation
        td = tdscf.TDA(mf)
        td.nstates = n_states
        td.kernel()

        # Extract results
        excitation_energies = td.e * 27.2114  # Convert to eV
        oscillator_strengths = td.oscillator_strength()

        # Return results in a dictionary
        return {
            'energies': excitation_energies,
            'oscillator_strengths': oscillator_strengths,
            'wavelengths': 1240 / excitation_energies,  # Convert eV to nm
            'transitions': td.transition_dipole()
        }

    def _build_molecule(self, geometry):
        """Build a PySCF molecule object from geometry.

        Args:
            geometry (str or dict): Molecular geometry

        Returns:
            pyscf.gto.Mole: PySCF molecule object
        """
        mol = gto.Mole()

        if isinstance(geometry, str):
            # Assume XYZ format string
            mol.atom = geometry
        elif isinstance(geometry, dict):
            # Convert dictionary format to PySCF format
            atom_list = []
            for atom, coords in geometry.items():
                if isinstance(coords[0], list):  # Multiple atoms of the same type
                    for coord in coords:
                        atom_list.append([atom, coord])
                else:  # Single atom
                    atom_list.append([atom, coords])
            mol.atom = atom_list

        # Set basis and other parameters from config
        mol.basis = self.config.get('basis', '6-31g')
        mol.verbose = self.config.get('verbose', 3)
        mol.max_memory = self.config.get('max_memory', 4000)  # MB

        mol.build()
        return mol
