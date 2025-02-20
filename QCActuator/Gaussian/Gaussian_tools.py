# gaussian_tools.py
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Union
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class GaussianConfig:
    """Configuration settings for Gaussian calculator"""
    work_dir: Optional[str] = None  # Working directory (None = use temp dir)
    verbose: bool = False  # Print additional information during calculations

@dataclass
class GaussianInput:
    """Data class for Gaussian input parameters"""
    xc: str
    basis: str
    nroot: int
    charge: int
    multiplicity: int
    geometry: str

class GaussianNotebookCalculator:
    def __init__(self, 
                 gaussian_path: str = "/work/home/huangm/source/gaussian_shared/",
                 config: Optional[GaussianConfig] = None):
        self.gaussian_path = Path(gaussian_path)
        self.gaussian_exec = str(self.gaussian_path / "g16")
        self.gaussian_formchk = str(self.gaussian_path / "formchk")
        self.config = config or GaussianConfig()
        self._working_dir = None
        self._spectrum_data = None
        self._current_title = "TD-DFT calculation"  # Default title
        
    def setup_working_dir(self) -> Path:
        """Set up working directory."""
        if self.config.work_dir:
            work_dir = Path(self.config.work_dir)
            work_dir.mkdir(parents=True, exist_ok=True)
            self._working_dir = work_dir
        else:
            self._working_dir = Path(tempfile.mkdtemp())
            
        if self.config.verbose:
            print(f"Working directory: {self._working_dir}")
            
        return self._working_dir

    def __enter__(self):
        self.setup_working_dir()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Don't clean up anything - keep all files
        pass

    def generate_input(self,
                      xc: str = "B3LYP",
                      basis: str = "6-31G",
                      nroot: int = 3,
                      charge: int = 0,
                      multiplicity: int = 1,
                      geometry: str = "",
                      title: str = "TD-DFT calculation",
                      extra_keywords: str = "",
                      freq: bool = False,
                      opt: bool = False) -> str:
        """
        Generate Gaussian input with comprehensive options.
        
        Args:
            xc: Exchange-correlation functional (e.g., B3LYP, PBE0)
            basis: Basis set (e.g., 6-31G, def2-SVP)
            nroot: Number of excited states for TD-DFT
            charge: Molecular charge
            multiplicity: Spin multiplicity
            geometry: Molecular geometry in XYZ format
            memory: Memory per core in MB
            nprocs: Number of processors
            extra_keywords: Additional Gaussian keywords
            solvent: Solvent name for PCM model
            freq: Add frequency calculation
            opt: Add geometry optimization
            
        Returns:
            str: Complete Gaussian input
        """
        # Ensure working directory is set up
        if not self._working_dir:
            self.setup_working_dir()
            
        # Create checkpoint file path in working directory
        chk_path = self._working_dir / "uv.chk"
            
        # Build keywords string
        route = f"# {xc}/{basis}"
        
        # Add optional keywords
        if opt:
            route += " Opt"
        if freq:
            route += " Freq"
        if nroot > 0:
            route += f" TD(NStates={nroot})"
        if extra_keywords:
            route += f" {extra_keywords}"
            
        # Combine everything
        input_str = f"%chk={str(chk_path)}\n"
        input_str += f"{route}\n\n"
        self._current_title = title  # Store the current title
        input_str += f"{title}\n\n"  # Title line
        input_str += f"{charge} {multiplicity}\n"
        input_str += f"{geometry.strip()}\n\n"
        
        if self.config.verbose:
            print("Generated Gaussian input:")
            print(input_str)
            
        return input_str

    def run_calculation(self, 
                       input_content: Union[str, GaussianInput], 
                       keep_files: Optional[bool] = None) -> Tuple[Path, Path]:
        """Run Gaussian calculation with input content."""
        if isinstance(input_content, GaussianInput):
            input_content = input_content.generate()
            
        if not self._working_dir:
            self.setup_working_dir()
            
        work_dir = Path(self._working_dir)
        inp_file = work_dir / "gaussian.gjf"
        out_file = work_dir / "gaussian.log"
        
        with open(inp_file, "w") as f:
            f.write(input_content)
            
        if self.config.verbose:
            print(f"Running Gaussian calculation in {work_dir}")
            
        with open(out_file, "w") as f_o:
            process = subprocess.run(
                [self.gaussian_exec, str(inp_file)],
                stdout=f_o,
                stderr=subprocess.PIPE,
                check=True
            )
            
        return inp_file, out_file

    def extract_excitations(self, output_file: Path) -> np.ndarray:
        """
        Extract excitation energies and oscillator strengths from Gaussian output.
        
        Args:
            output_file: Path to the Gaussian output file
            
        Returns:
            np.ndarray: Array of [energy (eV), oscillator strength] pairs
        """
        excitations = []
        
        with open(output_file, 'r') as f:
            lines = f.readlines()
            
            # Find the excitation section
            for i, line in enumerate(lines):
                if "Excitation energies and oscillator strengths:" in line:
                    # Start processing excited states
                    j = i + 1
                    while j < len(lines):
                        if "Excited State" in lines[j]:
                            # Parse the line for energy and oscillator strength
                            parts = lines[j].split()
                            energy = float(parts[4])  # Energy in eV
                            f_strength = float(parts[8].replace("f=", ""))
                            excitations.append([energy, f_strength])
                        elif "SavETr:" in lines[j]:  # End of excitation section
                            break
                        j += 1
                    break
        
        return np.array(excitations)

    def plot_spectrum(self, 
                     excitation_data: np.ndarray,
                     figsize: Tuple[int, int] = (10, 6),
                     title: Optional[str] = None,
                     xlabel: str = "Energy (eV)",
                     ylabel: str = "Oscillator Strength") -> None:
        """
        Create a stick plot of excitation energies and oscillator strengths.
        
        Args:
            excitation_data: Array of [energy, oscillator strength] pairs
            figsize: Figure size tuple (width, height)
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
        """
        plt.figure(figsize=figsize)
        
        # Create stick plot
        energies = excitation_data[:, 0]
        strengths = excitation_data[:, 1]
        
        plt.vlines(energies, 0, strengths, colors='b', linewidth=2)
        plt.plot(energies, strengths, 'bo', markersize=8)  # Add points at the tops
        
        # Customize plot
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title if title is not None else self._current_title)
        plt.grid(True, alpha=0.3)
        
        # Adjust limits to show full sticks with some padding
        plt.xlim(min(energies) - 0.5, max(energies) + 0.5)
        plt.ylim(0, max(strengths) * 1.1)
        
        plt.show()

    def save_working_dir(self, output_dir: Union[str, Path]) -> None:
        """Save entire working directory to a new location."""
        if not self._working_dir:
            raise ValueError("No working directory to save")
            
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy all files from working directory
        for file in Path(self._working_dir).glob("*"):
            with open(file, 'rb') as fsrc:
                with open(output_dir / file.name, 'wb') as fdst:
                    fdst.write(fsrc.read())
                    
        if self.config.verbose:
            print(f"Saved working directory contents to: {output_dir}")