# orca_tools.py
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Union
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class OrcaConfig:
    """Configuration settings for ORCA calculator"""
    work_dir: Optional[str] = None  # Working directory (None = use temp dir)
    verbose: bool = False  # Print additional information during calculations


@dataclass
class OrcaInput:
    """Data class for ORCA input parameters"""
    xc: str
    basis: str
    nroot: int
    charge: int
    multiplicity: int
    geometry: str


class OrcaNotebookCalculator:
    def __init__(
            self,
            orca_path: str = "/work/home/huangm/source/orca_6_0_0_shared_openmpi416/",
            config: Optional[OrcaConfig] = None):
        self.orca_path = Path(orca_path)
        self.orca_exec = str(self.orca_path / "orca")
        self.orca_mapspc = str(self.orca_path / "orca_mapspc")
        self.config = config or OrcaConfig()
        self._working_dir = None
        self._spectrum_data = None

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
                       extra_keywords: str = "",
                       freq: bool = False,
                       opt: bool = False) -> str:
        """
        Generate ORCA input with comprehensive options.

        Args:
            xc: Exchange-correlation functional (e.g., B3LYP, PBE0)
            basis: Basis set (e.g., 6-31G, def2-SVP)
            nroot: Number of excited states for TDDFT
            charge: Molecular charge
            multiplicity: Spin multiplicity
            geometry: Molecular geometry in XYZ format
            memory: Memory per core in MB
            nprocs: Number of processors
            extra_keywords: Additional ORCA keywords
            solvent: Solvent name for CPCM model
            freq: Add frequency calculation
            opt: Add geometry optimization

        Returns:
            str: Complete ORCA input
        """
        # Build main keywords
        keywords = [xc, f"{basis}*"]

        if opt:
            keywords.append("OPT")
        if freq:
            keywords.append("FREQ")
        if extra_keywords:
            keywords.append(extra_keywords)

        # Build input blocks
        blocks = []

        # TDDFT block
        blocks.append(f"%tddft\n  nroots {nroot}\nend")

        # Combine everything
        input_str = f"!{' '.join(keywords)}\n"
        input_str += "\n".join(blocks) + "\n"
        input_str += f"*xyz {charge} {multiplicity}\n"
        input_str += f"{geometry.strip()}\n"
        input_str += "*\n"

        if self.config.verbose:
            print("Generated ORCA input:")
            print(input_str)

        return input_str

    def run_calculation(self,
                        input_content: Union[str, OrcaInput],
                        keep_files: Optional[bool] = None) -> Tuple[Path, Path]:
        """Run ORCA calculation with input content."""
        if isinstance(input_content, OrcaInput):
            input_content = input_content.generate()

        if not self._working_dir:
            self.setup_working_dir()

        work_dir = Path(self._working_dir)
        inp_file = work_dir / "orca.inp"
        out_file = work_dir / "orca.out"

        with open(inp_file, "w") as f:
            f.write(input_content)

        if self.config.verbose:
            print(f"Running ORCA calculation in {work_dir}")

        with open(out_file, "w") as f_o:
            process = subprocess.run(
                [self.orca_exec, str(inp_file)],
                stdout=f_o,
                stderr=subprocess.PIPE,
                check=True
            )

        return inp_file, out_file

    def generate_spectrum(self,
                          output_file: Path,
                          type: str = "abs",
                          unit: str = "eV",
                          conv_width: float = 0.5,
                          start_point: float = 1.5,
                          end_point: float = 13.5) -> Dict[str, np.ndarray]:
        """Generate UV spectrum data from ORCA output."""
        subprocess.run([
            self.orca_mapspc,
            str(output_file),
            f"{type}",
            f"-{unit}",
            f"-x0{start_point}",
            f"-x1{end_point}",
            f"-w{conv_width}"
        ], check=True)

        stk_file = output_file.with_suffix('.out.ABS.stk')
        dat_file = output_file.with_suffix('.out.ABS.dat')

        # Load data into memory
        stk_data = np.loadtxt(stk_file)
        dat_data = np.loadtxt(dat_file)

        self._spectrum_data = {
            'stick': stk_data,
            'conv': dat_data
        }
        self._has_generated_spectrum = True

        return self._spectrum_data

    def plot_spectrum(self,
                      spectrum_data: Optional[Dict[str, np.ndarray]] = None,
                      figsize: Tuple[int, int] = (10, 6),
                      title: str = "UV-Vis Spectrum",
                      show_sticks: bool = True) -> None:
        """Plot UV-Vis spectrum."""
        data_to_plot = spectrum_data if spectrum_data is not None else self._spectrum_data

        if data_to_plot is None:
            raise ValueError(
                "No spectrum data available. Run generate_spectrum first.")

        plt.figure(figsize=figsize)

        conv_data = data_to_plot['conv']
        plt.plot(conv_data[:, 0], conv_data[:, 1], 'b-', label='Convoluted')

        if show_sticks:
            stick_data = data_to_plot['stick']
            plt.vlines(stick_data[:, 0], 0, stick_data[:, 1],
                       colors='r', linestyles='solid', alpha=0.5,
                       label='Stick')

        plt.xlabel('Energy (eV)')
        plt.ylabel('Intensity (arb. units)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
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
