import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Union, List
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum, auto


class SpectroscopyType(Enum):
    """
    Supported spectroscopy calculation types with mapping to spectrum generation parameters
    Each value contains (display_name, file_extension, default_unit)
    """
    UV = ("UV", "ABS", "eV")
    XAS = ("XAS", "ABS", "eV")
    NMR = ("NMR", "nmr", None)  # NMR uses a different approach
    IR = ("IR", "ir", "cm-1")
    RAMAN = ("Raman", "raman", "cm-1")
    VCD = ("VCD", "ir", "cm-1")
    ECD = ("ECD", "CD", "eV")

    def __init__(self, display_name, file_extension, default_unit):
        self.display_name = display_name
        self.file_extension = file_extension
        self.default_unit = default_unit


@dataclass
class OrcaConfig:
    """Configuration settings for ORCA calculator"""
    work_dir: Optional[str] = None  # Working directory (None = use temp dir)
    verbose: bool = False  # Print additional information during calculations


@dataclass
class OrcaInput:
    """Data class for ORCA input parameters"""
    xc: str = "B3LYP"
    basis: str = "6-31G*"
    nroot: int = 3
    charge: int = 0
    multiplicity: int = 1
    geometry: str = ""
    extra_keywords: str = ""
    solvent: Optional[str] = None
    core_orb: Optional[int] = 0
    memory: Optional[int] = None
    nprocs: Optional[int] = None
    # EFEI calculation parameters
    efei: bool = False
    efei_strength: Optional[float] = None
    efei_atoms: Optional[Tuple[int, int]] = None


class OrcaCalculator:
    def __init__(
            self,
            orca_path: str = "/usr/bin/orca",
            config: Optional[OrcaConfig] = None):
        self.orca_path = Path(orca_path)

        # Check if orca_path is a directory or the actual executable
        if self.orca_path.is_dir():
            self.orca_exec = str(self.orca_path / "orca")
            self.orca_mapspc = str(self.orca_path / "orca_mapspc")
            self.orca_nmrspec = str(
                self.orca_path / "orca_nmrspectrum")
        else:
            # If orca_path points directly to the executable
            self.orca_exec = str(self.orca_path)
            # Assume other tools are in the same directory
            self.orca_mapspc = str(self.orca_path.parent / "orca_mapspc")
            self.orca_nmrspec = str(
                self.orca_path.parent / "orca_nmrspectrum")

        self.config = config or OrcaConfig()
        self._working_dir = None
        self._spectrum_data = None
        self._has_generated_spectrum = False
        self._nmr_data = None  # Store NMR-specific data

        # Only print during initialization
        if self.config.verbose:
            print(f"Initialized with ORCA executable: {self.orca_exec}")
            print(f"ORCA mapspc tool: {self.orca_mapspc}")
            print(f"ORCA nmrspec tool: {self.orca_nmrspec}")

    def setup_working_dir(self) -> Path:
        """Set up a new working directory for each calculation."""
        if self.config.work_dir:
            # If a specific working directory is provided, create a unique subdirectory
            base_dir = Path(self.config.work_dir)
            # Create a timestamp-based unique directory name
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            work_dir = base_dir / f"orca_calc_{timestamp}"
            work_dir.mkdir(parents=True, exist_ok=True)
            self._working_dir = work_dir
        else:
            # Use a new temporary directory for each calculation
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

    def run_subprocess(self, args, check=True, capture_output=False):
        """
        Run a subprocess with controlled output based on verbose setting.   

        Args:
            args: List of command arguments
            check: Whether to check the return code
            capture_output: Whether to capture and return the output    

        Returns:
            CompletedProcess object from subprocess.run
        """
        if self.config.verbose:
            # In verbose mode, show the output
            if capture_output:
                # Capture output but also print it
                result = subprocess.run(
                    args, check=check, text=True, capture_output=True)
                print(f"STDOUT: {result.stdout}")
                if result.stderr:
                    print(f"STDERR: {result.stderr}")
                return result
            else:
                # Just run with output visible
                return subprocess.run(args, check=check)
        else:
            # In non-verbose mode, suppress output
            if capture_output:
                # Capture output silently
                return subprocess.run(args, check=check, text=True, capture_output=True)
            else:
                # Suppress all output
                with open(os.devnull, 'w') as devnull:
                    return subprocess.run(args, stdout=devnull, stderr=devnull, check=check)

    def generate_input(self,
                       xc: str = "B3LYP",
                       basis: str = "6-31G*",
                       nroot: int = 3,
                       charge: int = 0,
                       multiplicity: int = 1,
                       geometry: str = "",
                       extra_keywords: str = "",
                       freq: bool = False,
                       numfreq: bool = False,
                       tddft: bool = False,
                       opt: bool = False,
                       core: bool = False,
                       core_orb: int = 0,
                       dovcd: bool = False,
                       solvent: Optional[str] = None,
                       memory: Optional[int] = None,
                       nprocs: Optional[int] = None,
                       efei: bool = False,
                       efei_strength: Optional[float] = None,
                       efei_atoms: Optional[Tuple[int, int]] = None,
                       nmr_specific: bool = False,
                       xyz_file: Optional[Union[str, Path]] = None) -> str:
        """
        Generate ORCA input with comprehensive options.
        """
        # Build main keywords
        keywords = [xc, f"{basis}"]

        if opt:
            keywords.append("OPT")
        if freq:
            keywords.append("FREQ")
        if numfreq:
            keywords.append("NUMFREQ")
        if extra_keywords:
            keywords.append(extra_keywords)

        # Build input blocks
        blocks = []

        # TDDFT block
        if tddft:
            blocks.append(f"%tddft\n  nroots {nroot}")
            if core:
                blocks.append(f"orbwin[0] = {core_orb},{core_orb},-1,-1 \n")
            blocks.append(f"end")

        # Raman block:
        if numfreq:
            blocks.append(f"%ELPROP\n  POLAR 1 \nend")

        # VCD block:
        if dovcd:
            blocks.append(f"%FREQ\n  DOVCD true \nend")

        # Resource allocation
        resource_block = []
        if memory or nprocs:
            resource_block.append("%pal")
            if nprocs:
                resource_block.append(f"  nprocs {nprocs}")
            if memory:
                resource_block.append(f"  memoryPerCore {memory}")
            resource_block.append("end")
            blocks.append("\n".join(resource_block))

        # Solvent model
        if solvent:
            blocks.append(
                f"%cpcm\n  smd true\n  smdsolvent \"{solvent}\"\nend")

        # EFEI calculation
        if efei and efei_strength is not None and efei_atoms is not None:
            blocks.append(
                f"%geom\n   POTENTIALS\n     {{C {efei_atoms[0]} {efei_atoms[1]} {efei_strength}}} \n  end \nend")

        # Combine everything
        input_str = f"!{' '.join(keywords)}\n"
        input_str += "\n".join(blocks) + "\n"

        # Check if using external XYZ file or inline geometry
        if xyz_file is not None:
            xyz_path = Path(xyz_file)
            if not xyz_path.exists():
                raise FileNotFoundError(f"XYZ file not found: {xyz_path}")

            # Use xyzfile directive
            input_str += f"*xyzfile {charge} {multiplicity} \"{str(xyz_path)}\"\n"
        else:
            # Use inline geometry
            input_str += f"*xyz {charge} {multiplicity}\n"
            input_str += f"{geometry.strip()}\n"
            input_str += "*\n"

        # NMR specific block for EPRNMR with H and C shifts
        if nmr_specific:
            input_str += f"%eprnmr\n  Nuclei = all H {{SHIFT}}\n  Nuclei = all C {{SHIFT}} \nend"

        if self.config.verbose:
            print("Generated ORCA input:")
            print(input_str)

        return input_str

    def generate_input_for_spectroscopy(self,
                                        spec_type: SpectroscopyType,
                                        input_params: OrcaInput) -> str:
        """Generate input file specific to spectroscopy type"""
        # Set defaults
        freq = False
        numfreq = False
        opt = False
        extra = input_params.extra_keywords
        tddft = False
        nmr_specific = False
        core = False
        dovcd = False

        # Adjust parameters based on spectroscopy type
        if spec_type in [SpectroscopyType.UV, SpectroscopyType.ECD]:
            # UV, XAS and ECD need TDDFT but no FREQ/OPT
            tddft = True

        elif spec_type == SpectroscopyType.XAS:
            # XAS needs TDDFT but no FREQ/OPT
            tddft = True
            core = True

        elif spec_type == SpectroscopyType.IR:
            # IR needs FREQ but not NUMFREQ
            freq = True
            opt = True

        elif spec_type == SpectroscopyType.RAMAN:
            # Raman needs NUMFREQ
            freq = False
            numfreq = True
            opt = True

        elif spec_type == SpectroscopyType.VCD:
            # VCD needs both OPT and FREQ
            freq = True
            opt = True
            dovcd = True

        elif spec_type == SpectroscopyType.NMR:
            # NMR needs special keywords and EPRNMR block
            nmr_specific = True

        # Generate the input
        input_str = self.generate_input(
            xc=input_params.xc,
            basis=input_params.basis,
            nroot=input_params.nroot,
            charge=input_params.charge,
            multiplicity=input_params.multiplicity,
            geometry=input_params.geometry,
            extra_keywords=extra,
            freq=freq,
            numfreq=numfreq,
            tddft=tddft,
            core=core,
            core_orb=input_params.core_orb,
            opt=opt,
            dovcd=dovcd,
            solvent=input_params.solvent,
            memory=input_params.memory,
            nprocs=input_params.nprocs,
            efei=input_params.efei,
            efei_strength=input_params.efei_strength,
            efei_atoms=input_params.efei_atoms,
            nmr_specific=nmr_specific
        )

        return input_str

    def run_calculation(self,
                        input_content: Union[str, OrcaInput],
                        spec_type: Optional[SpectroscopyType] = None) -> Tuple[Path, Path]:
        """Run ORCA calculation with input content."""
        if isinstance(input_content, OrcaInput):
            if spec_type:
                input_content = self.generate_input_for_spectroscopy(
                    spec_type, input_content)
            else:
                input_content = self.generate_input(
                    xc=input_content.xc,
                    basis=input_content.basis,
                    nroot=input_content.nroot,
                    charge=input_content.charge,
                    multiplicity=input_content.multiplicity,
                    geometry=input_content.geometry,
                    extra_keywords=input_content.extra_keywords,
                    solvent=input_content.solvent,
                    core_orb=input_content.core_orb,
                    memory=input_content.memory,
                    nprocs=input_content.nprocs,
                    efei=input_content.efei,
                    efei_strength=input_content.efei_strength,
                    efei_atoms=input_content.efei_atoms
                )

        if not self._working_dir:
            self.setup_working_dir()

        work_dir = Path(self._working_dir)
        inp_file = work_dir / "orca.inp"
        out_file = work_dir / "orca.out"

        with open(inp_file, "w") as f:
            f.write(input_content)

        if self.config.verbose:
            print(f"Running ORCA calculation in {work_dir}")
            print(f"Using ORCA executable: {self.orca_exec}")

        # Verify ORCA executable exists
        if not Path(self.orca_exec).exists():
            raise FileNotFoundError(
                f"ORCA executable not found at: {self.orca_exec}")

        try:
            with open(out_file, "w") as f_o:
                process = subprocess.run(
                    [self.orca_exec, str(inp_file)],
                    stdout=f_o,
                    stderr=subprocess.PIPE,
                    check=True
                )

            if self.config.verbose:
                print(f"ORCA calculation completed successfully")

        except subprocess.CalledProcessError as e:
            print(f"ORCA calculation failed with error: {e}")
            print(f"Error output: {e.stderr.decode('utf-8')}")
            raise

        return inp_file, out_file

    def generate_nmr_spectrum(self,
                              output_file: Path,
                              nucleus: str = "1H",
                              line_width: float = 0.02,
                              spectrometer_freq: float = 400.0) -> Dict[str, np.ndarray]:
        """
        Generate NMR spectrum data using orca_nmrspectrum.

        Args:
            output_file: Path to the ORCA output file
            nucleus: Nucleus type ('1H' or '13C')
            line_width: Line width for the spectrum in Hz
            spectrometer_freq: Spectrometer frequency in MHz (default 400 MHz)

        Returns:
            Dict with spectrum data
        """
        # Check if orca_nmrspectrum exists
        orca_nmrspectrum = str(self.orca_path / "orca_nmrspectrum")
        if not Path(orca_nmrspectrum).exists():
            raise FileNotFoundError(
                f"ORCA nmrspectrum not found at: {orca_nmrspectrum}")

        # Get the base name without extension
        base_path = output_file.with_suffix('')
        gbw_file = base_path.with_suffix('.gbw')

        if not gbw_file.exists():
            raise FileNotFoundError(f"GBW file not found: {gbw_file}")

        # Create the nmrspec input file
        nmrspec_file = base_path.with_suffix('.nmrspec')
        with open(nmrspec_file, 'w') as f:
            f.write(
                f"NMRShieldingFile = \"{base_path}\"   #property file for shieldings\n")
            f.write(
                f"NMRCouplingFile  = \"None\"                        #property file for couplings\n")
            f.write(
                f"NMRSpecFreq = {spectrometer_freq:.2f}                       #spectrometer freq [MHz] (default 400)\n")
            f.write(
                f"PrintLevel = 1                            #PrintLevel for debugging info\n")
            f.write(
                f"NMRCoal = {line_width}                             #threshold for merged lines [Hz] (default 1)\n")
            f.write(
                f"END                                       #essential end of input\n")

        # Prepare output file for the nmrspectrum command
        nmrspec_output = base_path.with_suffix('.nmrspectrum.out')

        # Run orca_nmrspectrum command
        nmrspec_cmd = [
            orca_nmrspectrum,
            str(gbw_file),
            str(nmrspec_file)
        ]

        if self.config.verbose:
            print(
                f"Running NMR spectrum generation with: {' '.join(nmrspec_cmd)} > {nmrspec_output}")

        try:
            with open(nmrspec_output, 'w') as f_out:
                subprocess.run(nmrspec_cmd, stdout=f_out,
                               stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as e:
            print(f"NMR spectrum generation failed: {e}")
            print(f"Error output: {e.stderr.decode('utf-8')}")
            raise

        # Parse the output file to extract NMR peaks
        h_peaks = []  # For 1H
        c_peaks = []  # For 13C

        def adjust_shift(shift, ref_value, standard_ref):
            """Adjust chemical shift using reference values"""
            return standard_ref - ref_value + shift

        # Parse the output file
        with open(nmrspec_output, 'r') as f:
            content = f.read()

        # Extract sections for different atom types
        sections = []
        header_pattern = r"NMR Peaks for atom type (\d+), ref value\s+([0-9.-]+) ppm :"

        for match in re.finditer(header_pattern, content):
            atom_type = int(match.group(1))
            ref_value = float(match.group(2))

            # Find the start of the data (after "Atom   shift[ppm]   rel.intensity")
            section_start = content.find(
                "Atom   shift[ppm]   rel.intensity", match.end())
            if section_start == -1:
                print(f"Could not find data section for atom type {atom_type}")
                continue

            # Find the end of this section (next section or end of file)
            section_end = content.find(
                "NMR Peaks for atom type", section_start)
            if section_end == -1:
                section_end = len(content)

            # Extract the lines with the peak data
            data_text = content[section_start:section_end]
            # Skip the header line
            peaks_text = data_text.split(
                '\n', 1)[1] if '\n' in data_text else ""

            peaks = []
            for line in peaks_text.split('\n'):
                if line.strip() and not line.startswith('-'):
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        try:
                            atom = int(parts[0])
                            shift = float(parts[1])
                            rel_intensity = float(parts[2])
                            peaks.append({
                                'atom': atom,
                                'shift': shift,
                                'rel_intensity': rel_intensity
                            })
                        except (ValueError, IndexError) as e:
                            continue

            sections.append({
                'atom_type': atom_type,
                'ref_value': ref_value,
                'peaks': peaks
            })

        # Process sections and adjust chemical shifts
        for section in sections:
            atom_type = section['atom_type']
            ref_value = section['ref_value']
            peaks = section['peaks']

            if atom_type == 1:  # H
                standard_ref = 31.573
                for peak in peaks:
                    adjusted_shift = adjust_shift(
                        peak['shift'], ref_value, standard_ref)
                    h_peaks.append({
                        'shift': adjusted_shift,
                        'rel_intensity': peak['rel_intensity'],
                        'atom': peak['atom']
                    })
            elif atom_type == 6:  # C
                standard_ref = 193.023
                for peak in peaks:
                    adjusted_shift = adjust_shift(
                        peak['shift'], ref_value, standard_ref)
                    c_peaks.append({
                        'shift': adjusted_shift,
                        'rel_intensity': peak['rel_intensity'],
                        'atom': peak['atom']
                    })

        # Create spectrum data in the format expected by the rest of the code
        if nucleus == "1H" and h_peaks:
            # Sort peaks by chemical shift
            h_peaks.sort(key=lambda x: x['shift'])

            # Create stick data (compatible with existing stick data format)
            shifts = np.array([peak['shift'] for peak in h_peaks])
            intensities = np.array([peak['rel_intensity'] for peak in h_peaks])

            # Create stick data
            stick_data = np.column_stack((shifts, intensities))

            # Create convolution data (simplified for now)
            # Generate points along the spectrum range with a small step
            x_min = max(0, shifts.min() - 2)
            x_max = shifts.max() + 2
            x_points = np.linspace(x_min, x_max, 1000)

            # Apply Lorentzian broadening to each peak
            y_points = np.zeros_like(x_points)
            for shift, intensity in zip(shifts, intensities):
                # Simple Lorentzian function
                y_points += intensity * \
                    (0.5 * line_width) / \
                    ((x_points - shift)**2 + (0.5 * line_width)**2)

            # Create convolution data
            conv_data = np.column_stack((x_points, y_points))

            # Store spectrum data
            self._nmr_data = {
                'type': 'NMR',
                'nucleus': nucleus,
                'stick': stick_data,
                'conv': conv_data,
                'peaks': h_peaks
            }

        elif nucleus == "13C" and c_peaks:
            # Similar processing for carbon peaks
            c_peaks.sort(key=lambda x: x['shift'])

            shifts = np.array([peak['shift'] for peak in c_peaks])
            intensities = np.array([peak['rel_intensity'] for peak in c_peaks])

            stick_data = np.column_stack((shifts, intensities))

            x_min = max(0, shifts.min() - 10)
            x_max = shifts.max() + 10
            x_points = np.linspace(x_min, x_max, 1000)

            y_points = np.zeros_like(x_points)
            for shift, intensity in zip(shifts, intensities):
                # Carbon peaks are typically sharper
                y_points += intensity * \
                    (0.5 * line_width) / \
                    ((x_points - shift)**2 + (0.5 * line_width)**2)

            conv_data = np.column_stack((x_points, y_points))

            self._nmr_data = {
                'type': 'NMR',
                'nucleus': nucleus,
                'stick': stick_data,
                'conv': conv_data,
                'peaks': c_peaks
            }
        else:
            # If no peaks found for the requested nucleus
            self._nmr_data = {
                'type': 'NMR',
                'nucleus': nucleus,
                'stick': np.array([[0, 0]]),  # Empty data
                'conv': np.array([[0, 0], [10, 0]]),  # Empty spectrum
                'peaks': []
            }

        self._has_generated_spectrum = True
        return self._nmr_data

    def generate_spectrum(self,
                          output_file: Path,
                          spec_type: SpectroscopyType,
                          unit: Optional[str] = None,
                          conv_width: float = 0.5,
                          start_point: float = 1.5,
                          end_point: float = 13.5) -> Dict[str, np.ndarray]:
        """Generate spectrum data from ORCA output."""

        # Handle NMR special case
        if spec_type == SpectroscopyType.NMR:
            # For NMR, use the specialized NMR spectrum generator
            h_spectrum = self.generate_nmr_spectrum(
                output_file, nucleus="1H")
            c_spectrum = self.generate_nmr_spectrum(
                output_file, nucleus="13C")

            # Return the H spectrum by default, but keep both
            self._spectrum_data = {
                'type': 'NMR',
                '1H': h_spectrum,
                '13C': c_spectrum,
                'current': h_spectrum  # Default to showing proton NMR
            }
            return self._spectrum_data

        # Get file extension from the enum
        file_extension = spec_type.file_extension

        # Use default unit from enum if not specified
        if unit is None:
            unit = spec_type.default_unit

        # Adjust parameters for orca_mapspc
        mapspc_args = [
            self.orca_mapspc,
            str(output_file),
            file_extension,  # Use the file extension as the spectrum type
            f"-{unit}"
        ]

        # Add range parameters where applicable for UV/XAS/ECD
        if spec_type in [SpectroscopyType.UV, SpectroscopyType.XAS, SpectroscopyType.ECD]:
            mapspc_args.extend([
                f"-x0{start_point}",
                f"-x1{end_point}",
                f"-w{conv_width}"
            ])

        if self.config.verbose:
            print(f"Running spectrum generation with: {' '.join(mapspc_args)}")

        self.run_subprocess(mapspc_args, check=True)

        # Get file extension for the output files
        extension = file_extension

        stk_file = output_file.with_suffix(f'.out.{extension}.stk')
        dat_file = output_file.with_suffix(f'.out.{extension}.dat')

        # Check if files exist
        if not stk_file.exists() or not dat_file.exists():
            raise FileNotFoundError(
                f"Spectrum files not found: {stk_file} or {dat_file}")

        # Load data into memory
        stk_data = np.loadtxt(stk_file)
        dat_data = np.loadtxt(dat_file)

        self._spectrum_data = {
            'stick': stk_data,
            'conv': dat_data,
            'type': spec_type.display_name
        }
        self._has_generated_spectrum = True

        return self._spectrum_data

    def plot_spectrum(self,
                      spectrum_data: Optional[Dict[str, np.ndarray]] = None,
                      figsize: Tuple[int, int] = (10, 6),
                      title: Optional[str] = None,
                      show_sticks: bool = True,
                      x_label: Optional[str] = None,
                      y_label: Optional[str] = None,
                      nucleus: Optional[str] = None) -> None:
        """Plot spectrum."""

        data_to_plot = spectrum_data if spectrum_data is not None else self._spectrum_data

        if data_to_plot is None:
            raise ValueError(
                "No spectrum data available. Run generate_spectrum first.")

        # Special handling for NMR data
        if data_to_plot.get('type') == 'NMR':
            if nucleus:
                # Use specified nucleus
                if nucleus not in ['1H', '13C']:
                    raise ValueError("NMR nucleus must be '1H' or '13C'")
                nmr_data = data_to_plot.get(nucleus)
            else:
                # Use current nucleus (default to 1H)
                nmr_data = data_to_plot.get('current', data_to_plot.get('1H'))
                nucleus = '1H' if 'current' not in data_to_plot else data_to_plot.get(
                    'current', {}).get('nucleus', '1H')

            if not nmr_data:
                raise ValueError(f"NMR data for {nucleus} not found")

            # Set NMR-specific defaults
            spec_type = "NMR"
            if not title:
                title = f"{nucleus} NMR Spectrum"
            if not x_label:
                x_label = "Chemical Shift (ppm)"
            if not y_label:
                y_label = "Signal Intensity"

            # Plot NMR data
            plt.figure(figsize=figsize)

            # For NMR, x-axis is usually reversed
            plt.gca().invert_xaxis()

            # Plot convoluted spectrum
            plt.plot(nmr_data['conv'][:, 0], nmr_data['conv']
                     [:, 1], 'b-', label='Convoluted')

            # Plot stick spectrum if requested
            if show_sticks and 'stick' in nmr_data:
                plt.vlines(nmr_data['stick'][:, 0], 0, nmr_data['stick'][:, 1],
                           colors='r', linestyles='solid', alpha=0.5,
                           label='Stick')
        else:
            # Regular spectrum handling
            spec_type = data_to_plot.get('type', 'UV')

            conv_data = data_to_plot['conv']
            # Invert Y-axis for IR, Raman, VCD
            if spec_type in ['IR']:
                conv_data[:, 1] = 1000 - conv_data[:, 1]
            plt.figure(figsize=figsize)
            plt.plot(conv_data[:, 0], conv_data[:, 1],
                     'b-', label='Convoluted')

            if show_sticks and 'stick' in data_to_plot:
                stick_data = data_to_plot['stick']
                plt.vlines(stick_data[:, 0], 0, stick_data[:, 1],
                           colors='r', linestyles='solid', alpha=0.5,
                           label='Stick')

            if not x_label:
                if spec_type in ['UV', 'XAS', 'ECD']:
                    x_label = 'Energy (eV)'
                elif spec_type in ['IR', 'Raman', 'VCD']:
                    x_label = 'Wavenumber (cm⁻¹)'
                elif spec_type == 'NMR':
                    x_label = 'Chemical Shift (ppm)'
                else:
                    x_label = 'X-axis'

            if not y_label:
                if spec_type in ['UV', 'XAS']:
                    y_label = 'Absorption (arb. units)'
                elif spec_type in ['IR', 'Raman']:
                    y_label = 'Intensity (arb. units)'
                elif spec_type in ['VCD', 'ECD']:
                    y_label = 'Rotational Strength (arb. units)'
                elif spec_type == 'NMR':
                    y_label = 'Signal Intensity'
                else:
                    y_label = 'Y-axis'

            if not title:
                title_map = {
                    'UV': 'UV-Vis Absorption Spectrum',
                    'XAS': 'X-ray Absorption Spectrum',
                    'IR': 'Infrared Spectrum',
                    'Raman': 'Raman Spectrum',
                    'VCD': 'Vibrational Circular Dichroism Spectrum',
                    'ECD': 'Electronic Circular Dichroism Spectrum',
                    'NMR': 'Nuclear Magnetic Resonance Spectrum',
                }
                title = title_map.get(spec_type, f'{spec_type} Spectrum')

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_nmr_comparison(self, figsize: Tuple[int, int] = (12, 8)):
        """Plot both 1H and 13C NMR spectra in a single figure for comparison."""
        if not self._nmr_data or 'type' not in self._nmr_data or self._nmr_data['type'] != 'NMR':
            raise ValueError(
                "NMR data not available. Run generate_spectrum with NMR type first.")

        if '1H' not in self._spectrum_data or '13C' not in self._spectrum_data:
            raise ValueError(
                "Both 1H and 13C NMR data required for comparison.")

        h_data = self._spectrum_data['1H']
        c_data = self._spectrum_data['13C']

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        # Plot 1H NMR on the first subplot
        ax1.plot(h_data['conv'][:, 0], h_data['conv'][:, 1], 'b-')
        ax1.vlines(h_data['stick'][:, 0], 0, h_data['stick'][:, 1],
                   colors='r', linestyles='solid', alpha=0.5)
        ax1.invert_xaxis()
        ax1.set_title('¹H NMR Spectrum')
        ax1.set_xlabel('Chemical Shift (ppm)')
        ax1.set_ylabel('Signal Intensity')
        ax1.grid(True, alpha=0.3)

        # Plot 13C NMR on the second subplot
        ax2.plot(c_data['conv'][:, 0], c_data['conv'][:, 1], 'g-')
        ax2.vlines(c_data['stick'][:, 0], 0, c_data['stick'][:, 1],
                   colors='r', linestyles='solid', alpha=0.5)
        ax2.invert_xaxis()
        ax2.set_title('¹³C NMR Spectrum')
        ax2.set_xlabel('Chemical Shift (ppm)')
        ax2.set_ylabel('Signal Intensity')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # Add helper function to visualize XYZ files

    def visualize_xyz_file(xyz_file, style="stick", surface=None, spin=False, calculator=None):
        """
        Convenience function to visualize an XYZ file without running calculations.

        Args:
            xyz_file: Path to XYZ file
            style: Visualization style ('stick', 'line', 'sphere', 'cartoon')
            surface: Surface type (None, 'VDW', 'MS', 'SAS', 'SES')
            spin: Whether to spin the molecule
            calculator: Existing OrcaCalculator instance (to reuse paths)
        """
        if calculator is None:
            calc = OrcaCalculator()
        else:
            calc = calculator

        calc.visualize_molecule(
            xyz_file=xyz_file, style=style, surface=surface, spin=spin)

    def run_spectroscopy(self,
                         geometry: str,
                         spec_type: str,
                         xc: str = "B3LYP",
                         basis: str = "6-31G*",
                         nroot: int = 3,
                         charge: int = 0,
                         multiplicity: int = 1,
                         extra_keywords: str = "",
                         solvent: Optional[str] = None,
                         core_orb: int = 0,
                         efei: bool = False,
                         efei_strength: Optional[float] = None,
                         efei_atoms: Optional[Tuple[int, int]] = None,
                         **kwargs) -> Dict[str, np.ndarray]:
        """
        Run a complete spectroscopy workflow: prepare input, run calculation, and generate spectrum.
        Uses a new working directory for each run.
        """
        # Ensure we start with a new working directory
        self.setup_working_dir()

        def _convert_string_to_spec_type(spec_type_str: str) -> SpectroscopyType:
            """
             Convert a string to the corresponding SpectroscopyType enum value.
             """
            spec_type_upper = spec_type_str.upper()
            try:
                return SpectroscopyType[spec_type_upper]
            except KeyError:
                raise ValueError(f"Invalid spectroscopy type: {spec_type_str}. "
                                 f"Valid types are: {', '.join([t.name.lower() for t in SpectroscopyType])}")

        # Convert string to SpectroscopyType
        spec_type_enum = _convert_string_to_spec_type(spec_type)

        # Prepare input parameters
        input_params = OrcaInput(
            xc=xc,
            basis=basis,
            nroot=nroot,
            charge=charge,
            multiplicity=multiplicity,
            geometry=geometry,
            extra_keywords=extra_keywords,
            solvent=solvent,
            core_orb=core_orb,
            efei=efei,
            efei_strength=efei_strength,
            efei_atoms=efei_atoms
        )

        # Run calculation
        _, out_file = self.run_calculation(
            input_params, spec_type=spec_type_enum)

        # Special handling for VCD spectra
        if spec_type_enum == SpectroscopyType.VCD:
            spectrum_data = self.extract_vcd_data(out_file)
            self.plot_vcd_spectrum(spectrum_data)
        else:
            # Generate spectrum with default unit from the spec_type
            spectrum_data = self.generate_spectrum(
                out_file,
                spec_type=spec_type_enum,
                **kwargs
            )
            # Plot spectrum using the standard method
            self.plot_spectrum(spectrum_data)

        # For spectroscopy types that include geometry optimization
        if spec_type_enum in [SpectroscopyType.IR, SpectroscopyType.RAMAN, SpectroscopyType.VCD]:
            # Look for the orca.xyz file which is automatically generated by ORCA
            xyz_file = Path(self._working_dir) / "orca.xyz"

            if xyz_file.exists():
                # Visualize the optimized structure if py3Dmol is available
                try:
                    self.visualize_molecule(xyz_file=xyz_file)
                    if self.config.verbose:
                        print(f"Visualized optimized geometry from {xyz_file}")
                except Exception as e:
                    if self.config.verbose:
                        print(f"Failed to visualize optimized geometry: {e}")

        return spectrum_data

    def save_geometry_to_xyz(self, geometry: str, filename: Optional[str] = None) -> Path:
        """
        Save molecular geometry to an XYZ file.
        """
        if not self._working_dir:
            self.setup_working_dir()

        # Clean the geometry
        lines = geometry.strip().split('\n')
        atom_count = len(lines)

        # Create proper XYZ format (with atom count and comment line)
        xyz_content = f"{atom_count}\n"
        xyz_content += "Generated by OrcaCalculator\n"
        xyz_content += "\n".join(lines)

        # Determine output path
        if filename is None:
            output_path = Path(self._working_dir) / "molecule.xyz"
        else:
            output_path = Path(filename)

        # Write to file
        with open(output_path, "w") as f:
            f.write(xyz_content)

        if self.config.verbose:
            print(f"Saved geometry to XYZ file: {output_path}")

        return output_path

    def visualize_molecule(self, geometry: Optional[str] = None, xyz_file: Optional[Union[str, Path]] = None,
                           width: int = 600, height: int = 400, style: str = "stick",
                           surface: Optional[str] = None, spin: bool = False) -> None:
        """
        Visualize a molecule using py3Dmol (if available).
        """
        try:
            import py3Dmol
            from IPython.display import display
        except ImportError:
            raise ImportError(
                "py3Dmol is not installed. Install it with: pip install py3Dmol")

        # Get the XYZ file path
        if geometry is not None:
            xyz_path = self.save_geometry_to_xyz(geometry)
        elif xyz_file is not None:
            xyz_path = Path(xyz_file)
            if not xyz_path.exists():
                raise FileNotFoundError(f"XYZ file not found: {xyz_path}")
        else:
            raise ValueError("Either geometry or xyz_file must be provided")

        # Read XYZ file
        with open(xyz_path, "r") as f:
            xyz_data = f.read()

        # Create viewer
        viewer = py3Dmol.view(width=width, height=height)
        viewer.addModel(xyz_data, "xyz")

        # Set style
        if style == "stick":
            viewer.setStyle({"stick": {}})
        elif style == "line":
            viewer.setStyle({"line": {}})
        elif style == "sphere":
            viewer.setStyle({"sphere": {}})
        elif style == "cartoon":
            viewer.setStyle({"cartoon": {}})
        else:
            viewer.setStyle({"stick": {}})

        # Add surface if requested
        if surface:
            viewer.addSurface(py3Dmol.VDW, {"opacity": 0.7, "color": "white"})

        # Set camera and spin
        viewer.zoomTo()
        if spin:
            viewer.spin(True)

        # Display
        display(viewer.show())

    def extract_vcd_data(self, output_file: Path) -> Dict[str, np.ndarray]:
        """
        Extract VCD data directly from ORCA output file.

        Args:
            output_file: Path to the ORCA output file

        Returns:
            Dict with extracted VCD data
        """
        if not output_file.exists():
            raise FileNotFoundError(f"Output file not found: {output_file}")

        # Read the output file
        with open(output_file, "r") as f:
            content = f.read()

        # Find the VCD spectrum section
        vcd_section_start = content.find("VCD SPECTRUM CALCULATION")
        if vcd_section_start == -1:
            raise ValueError("VCD spectrum data not found in output file")

        # Find the data table - match the header line more precisely
        table_marker = " Mode   Freq    VCD-Intensity"
        table_header = content.find(table_marker, vcd_section_start)
        if table_header == -1:
            raise ValueError("VCD data table not found in output file")

        # Find the dashed line below the header
        dash_line = content.find(
            "---------------------------------", table_header + len(table_marker))
        if dash_line == -1:
            raise ValueError("VCD data table structure not as expected")

        # The data starts after the dash line
        table_start = content.find("\n", dash_line) + 1

        # Find the end of the data - either another dash line or a blank line followed by a new section
        next_dash = content.find(
            "---------------------------------", table_start)
        next_blank = content.find("\n\n", table_start)

        if next_dash != -1 and (next_blank == -1 or next_dash < next_blank):
            table_end = next_dash
        elif next_blank != -1:
            table_end = next_blank
        else:
            table_end = len(content)

        # Parse the data lines
        data_lines = content[table_start:table_end].strip().split("\n")

        # Extract mode numbers, frequencies, and VCD intensities
        modes = []
        frequencies = []
        intensities = []

        for line in data_lines:
            line = line.strip()
            if not line:
                continue

            # Split by whitespace and handle variable whitespace
            parts = line.split()
            if len(parts) >= 3:
                try:
                    mode = int(parts[0])
                    freq = float(parts[1])
                    intensity = float(parts[2])

                    modes.append(mode)
                    frequencies.append(freq)
                    intensities.append(intensity)
                except (ValueError, IndexError) as e:
                    print(f"Skipping line due to error: {e} - Line: {line}")
                    continue

        # Convert to numpy arrays
        modes = np.array(modes)
        frequencies = np.array(frequencies)
        intensities = np.array(intensities)

        # Create stick data (compatible with existing stick data format)
        stick_data = np.column_stack((frequencies, intensities))

        # Store the extracted data
        vcd_data = {
            'type': 'VCD',
            'modes': modes,
            'stick': stick_data,
            # No convolution data since we're not using orca_mapspc
            'raw_intensities': intensities
        }

        # Set the spectrum data
        self._spectrum_data = vcd_data
        self._has_generated_spectrum = True

        return vcd_data

    def plot_vcd_spectrum(self,
                          vcd_data: Optional[Dict[str, np.ndarray]] = None,
                          figsize: Tuple[int, int] = (10, 6),
                          title: str = "Vibrational Circular Dichroism Spectrum",
                          line_color: str = "blue",
                          stick_color: str = "red",
                          lorentzian_width: float = 8.0,
                          show_grid: bool = True) -> None:
        """
        Plot VCD spectrum from extracted data with Lorentzian convolution.

        Args:
            vcd_data: Dictionary containing VCD data
            figsize: Figure size
            title: Plot title
            line_color: Color for the convolution line
            stick_color: Color for the sticks
            lorentzian_width: Width parameter for Lorentzian function (in wavenumbers)
            show_grid: Whether to show grid lines
        """
        data = vcd_data if vcd_data is not None else self._spectrum_data

        if data is None or data.get('type') != 'VCD':
            raise ValueError(
                "No VCD data available. Run extract_vcd_data first.")

        # Create figure
        plt.figure(figsize=figsize)

        # Extract data
        frequencies = data['stick'][:, 0]
        intensities = data['stick'][:, 1]

        # Plot stick spectrum (without markers at the top)
        plt.vlines(frequencies, 0, intensities, colors=stick_color, linestyles='solid',
                   label='VCD Intensity')

        # Generate x values for the convolution (higher resolution)
        # Prevent negative frequencies
        min_freq = max(np.min(frequencies) - 100, 0)
        max_freq = np.max(frequencies) + 100
        x_values = np.linspace(min_freq, max_freq, 1000)

        # Apply Lorentzian convolution
        y_values = np.zeros_like(x_values)
        for freq, intens in zip(frequencies, intensities):
            # Lorentzian function: f(x) = (1/π) * (γ/2) / ((x-x₀)² + (γ/2)²)
            # Where γ is the width parameter (FWHM)
            y_values += intens * (lorentzian_width/2)**2 / \
                ((x_values - freq)**2 + (lorentzian_width/2)**2)

        # Plot the convolution spectrum
        plt.plot(x_values, y_values, color=line_color, linewidth=2,
                 label=f'Convoluted')

        # Set labels and title
        plt.xlabel('Frequency (cm⁻¹)')
        plt.ylabel('VCD Intensity (1E-44*esu²*cm²)')
        plt.title(title)

        # Show grid if requested
        if show_grid:
            plt.grid(True, alpha=0.3)

        # Add legend
        plt.legend()

        # Show plot
        plt.show()

        return None  # Return the figure for further customization if needed

    def run_vcd_analysis(self,
                         geometry: str,
                         xc: str = "B3LYP",
                         basis: str = "6-31G*",
                         charge: int = 0,
                         multiplicity: int = 1,
                         extra_keywords: str = "",
                         solvent: Optional[str] = None,
                         memory: Optional[int] = None,
                         nprocs: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Run a complete VCD workflow: prepare input, run calculation, and extract VCD data.

        Args:
            geometry: Molecular geometry in XYZ format
            xc: Exchange-correlation functional
            basis: Basis set
            charge: Molecular charge
            multiplicity: Spin multiplicity
            extra_keywords: Additional ORCA keywords
            solvent: Solvent name for CPCM model
            memory: Memory per core in MB
            nprocs: Number of processors

        Returns:
            Dict[str, np.ndarray]: VCD spectrum data
        """
        # Prepare input parameters for VCD
        input_params = OrcaInput(
            xc=xc,
            basis=basis,
            charge=charge,
            multiplicity=multiplicity,
            geometry=geometry,
            extra_keywords=extra_keywords,
            solvent=solvent,
            memory=memory,
            nprocs=nprocs
        )

        # Run calculation with VCD settings
        _, out_file = self.run_calculation(
            input_params, spec_type=SpectroscopyType.VCD)

        # Extract VCD data directly from output
        vcd_data = self.extract_vcd_data(out_file)

        # Plot VCD spectrum
        self.plot_vcd_spectrum(vcd_data)

        return vcd_data

    def run_efei_calculation(self,
                             geometry: str,
                             efei_atoms: Tuple[int, int],
                             efei_strength: float,
                             xc: str = "B3LYP",
                             basis: str = "6-31G*",
                             charge: int = 0,
                             multiplicity: int = 1,
                             extra_keywords: str = "",
                             solvent: Optional[str] = None,
                             memory: Optional[int] = None,
                             nprocs: Optional[int] = None) -> Dict:
        """
        Run an EFEI (Electric Field Effect) geometry optimization calculation.

        Args:
            geometry: Molecular geometry in XYZ format
            efei_atoms: Tuple of atom indices for EFEI field direction (start, end)
            efei_strength: Field strength for EFEI calculation (in a.u.)
            xc: Exchange-correlation functional
            basis: Basis set
            charge: Molecular charge
            multiplicity: Spin multiplicity
            extra_keywords: Additional ORCA keywords
            solvent: Solvent name for CPCM model
            memory: Memory per core in MB
            nprocs: Number of processors

        Returns:
            Dict: Information about the calculation including paths to input/output files
        """
        print(
            f"Running EFEI geometry optimization with field strength {efei_strength} between atoms {efei_atoms}")

        # Ensure we have the correct required parameters
        if not all([efei_atoms, efei_strength is not None]):
            raise ValueError(
                "EFEI calculation requires both efei_atoms and efei_strength")

        # Save initial geometry to XYZ for visualization
        init_xyz_path = self.save_geometry_to_xyz(geometry,
                                                  filename=str(
                                                      Path(self._working_dir) / "initial.xyz")
                                                  if self._working_dir else None)

        # Visualize the initial structure
        try:
            print("Visualizing initial geometry...")
            self.visualize_molecule(xyz_file=init_xyz_path)
        except Exception as e:
            print(f"Failed to visualize initial geometry: {e}")

        # Prepare input parameters with EFEI specific settings for geometry optimization
        input_params = OrcaInput(
            xc=xc,
            basis=basis,
            charge=charge,
            multiplicity=multiplicity,
            geometry=geometry,
            extra_keywords=f"OPT {extra_keywords}".strip(),  # Add OPT keyword
            solvent=solvent,
            memory=memory,
            nprocs=nprocs,
            efei=True,
            efei_strength=efei_strength,
            efei_atoms=efei_atoms
        )

        # Run the calculation (no spectroscopy type)
        inp_file, out_file = self.run_calculation(input_params)

        # Look for optimized geometry
        xyz_file = Path(self._working_dir) / "orca.xyz"

        result = {
            'input_file': str(inp_file),
            'output_file': str(out_file),
            'initial_xyz': str(init_xyz_path),
            'optimized_xyz': str(xyz_file) if xyz_file.exists() else None,
            'efei': {
                'strength': efei_strength,
                'atoms': efei_atoms
            }
        }

        # Visualize the optimized geometry if available
        if xyz_file.exists():
            try:
                print(f"Visualizing optimized geometry from {xyz_file}")
                self.visualize_molecule(xyz_file=xyz_file)
                result['success'] = True
            except Exception as e:
                print(f"Failed to visualize optimized geometry: {e}")
        else:
            if self.config.verbose:
                print("No optimized geometry found. Check output file for errors.")
            result['success'] = False

        return result
