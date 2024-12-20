from numpy import *
import os
import sys
from scipy import special
import matplotlib.pyplot as plt

###########################################################
# Note: Plot multiple spetra with different modes and
#       convolution schemes:
#       Lorenzian, Gaussian, or Vogit
# Versopm: 0.2
# Author: Yu Zhang
# Date: Dec. 2024
#
###########################################################

# Plotting XAS signals
def ConvPlot_XAS(gamma, PlotTitle, PlotLabel, profile=1, normalized=0,
                 EShift=0.0, nsets=1,
                 PlotType=1, WithSticks=1, EnergyMargin=2.0,
                 ExptDataFile="", DataFileName="simu_data.dat", NoPoints=1000,
                 XLabel="Energy (eV)", YLabel="Oscillation Strength (a. u.)",
                 SaveFile=False, SaveFileType=2, SaveFileName="plot", ScaleStick=1.0):

    # Read the data file: energy vs. oscillation strength
    data = genfromtxt(DataFileName, dtype='float')

    dim = len(data[:, 0])
    Emin = amin(data[:, 0])
    Emax = amax(data[:, 0])

# Build a 1D grid for plotting
    Estep = (Emax - Emin + 2.0*EnergyMargin)/NoPoints
    Egrid = arange(Emin-EnergyMargin, Emax+EnergyMargin, Estep)

# Creat a list of zeros
    z = zeros(len(Egrid))
    if profile == 4:
        z1 = zeros(len(Egrid))
        z2 = zeros(len(Egrid))
        z3 = zeros(len(Egrid))

# Sum on the grid
    for i in range(dim):
        if profile == 1:
            z = z + data[i, 1]*1.0/pi*gamma/2.0 / \
                ((Egrid-data[i, 0])**2+gamma*gamma/4.0)
        elif profile == 2:
            sigma = gamma/(2.0*sqrt(2.0*log(2.0)))
            z = z + data[i, 1]*exp(-0.5*((Egrid-data[i, 0])/sigma)
                               ** 2)/(sigma*sqrt(2.0*pi))
        elif profile == 3:
            zz = (Egrid-data[i, 0]+1j*gamma/2.0)/(sigma*sqrt(2.0))
#		wz = exp(-1.0*zz*zz)*special.erfc(-1j*zz)
# wofz: the SciPy implementation of the Faddeeva function
            z = z + data[i, 1]*special.wofz(zz).real/(sigma*sqrt(2.0*pi))
        elif profile == 4:
            z1 = z1 + data[i, 1]*1.0/pi*gamma/2.0 / \
                ((Egrid-data[i, 0])**2+gamma*gamma/4.0)
            z2 = z2 + \
                data[i, 1]*exp(-0.5*((Egrid-data[i, 0])/sigma)
                               ** 2)/(sigma*sqrt(2.0*pi))
            zz = (Egrid-data[i, 0]+1j*gamma/2.0)/(sigma*sqrt(2.0))
#                wz = exp(-1.0*zz*zz)*special.erfc(-1j*zz)
            z3 = z3 + data[i, 1]*special.wofz(zz).real/(sigma*sqrt(2.0*pi))
        else:
            print("Wrong lineshape option!")
            sys.exit(1)

    fig, ax1 = plt.subplots()
    if profile != 4:
        plt.plot(Egrid, z, label=PlotLabel)
    else:
        plt.plot(Egrid, z1, label="Lorentzian")
        plt.plot(Egrid, z2, label="Gaussian")
        plt.plot(Egrid, z3, label="Voigt")

# No sticks if overlapped plots
    if (nsets > 1) and (PlotType == 1):
        WithSticks = 0

    if WithSticks:
        plt.vlines(data[:, 0], [0], ScaleStick*data[:, 1], colors='black')

    plt.legend()
    plt.xlabel(XLabel)
    plt.ylabel(YLabel)
    plt.title(PlotTitle)
    ax1.xaxis.set_tick_params(direction='in', top='on', which='both')
    ax1.yaxis.set_tick_params(direction='in', top='on', which='both')
    plt.show()

    if SaveFile:
        if SaveFileType == 1:
            plt.savefig(SaveFileName+'.svg')
        elif SaveFileType == 2:
            plt.savefig(SaveFileName+'.pdf')
        elif SaveFileType == 3:
            plt.savefig(SaveFileName+'.png')
        else:
            print("Wrong SaveFileType!")
            sys.exit(1)

# Plotting UV-vis signals
def ConvPlot_UVvis(gamma, PlotTitle, PlotLabel, profile=1, normalized=0,
                   EShift=0.0, nsets=1,
                   PlotType=1, WithSticks=1, EnergyMargin=0.1,
                   ExptDataFile="", DataFileName="simu_data.dat", NoPoints=1000,
                   XLabel="Energy (nm)", YLabel="Oscillation Strength (a. u.)",
                   SaveFile=False, SaveFileType=2, SaveFileName="plot", ScaleStick=1.0):

    # Read the data file: energy vs. oscillation strength
    data = genfromtxt(DataFileName, dtype='float')

    dim = len(data[:, 0])
    Emin = amin(data[:, 0])
    Emax = amax(data[:, 0])

# Build a 1D grid for plotting
    Estep = (Emax - Emin + 2.0*EnergyMargin)/NoPoints
    Egrid = arange(Emin-EnergyMargin, Emax+EnergyMargin, Estep)

# Creat a list of zeros
    z = zeros(len(Egrid))
    if profile == 4:
        z1 = zeros(len(Egrid))
        z2 = zeros(len(Egrid))
        z3 = zeros(len(Egrid))

# Sum on the grid
    for i in range(dim):
        if profile == 1:
            z = z + data[i, 1]*1.0/pi*gamma/2.0 / \
                ((Egrid-data[i, 0])**2+gamma*gamma/4.0)
        elif profile == 2:
            sigma = gamma/(2.0*sqrt(2.0*log(2.0)))
            z = z + data[i, 1]*exp(-0.5*((Egrid-data[i, 0])/sigma)
                               ** 2)/(sigma*sqrt(2.0*pi))
        elif profile == 3:
            zz = (Egrid-data[i, 0]+1j*gamma/2.0)/(sigma*sqrt(2.0))
#		wz = exp(-1.0*zz*zz)*special.erfc(-1j*zz)
# wofz: the SciPy implementation of the Faddeeva function
            z = z + data[i, 1]*special.wofz(zz).real/(sigma*sqrt(2.0*pi))
        elif profile == 4:
            z1 = z1 + data[i, 1]*1.0/pi*gamma/2.0 / \
                ((Egrid-data[i, 0])**2+gamma*gamma/4.0)
            z2 = z2 + \
                data[i, 1]*exp(-0.5*((Egrid-data[i, 0])/sigma)
                               ** 2)/(sigma*sqrt(2.0*pi))
            zz = (Egrid-data[i, 0]+1j*gamma/2.0)/(sigma*sqrt(2.0))
#                wz = exp(-1.0*zz*zz)*special.erfc(-1j*zz)
            z3 = z3 + data[i, 1]*special.wofz(zz).real/(sigma*sqrt(2.0*pi))
        else:
            print("Wrong lineshape option!")
            sys.exit(1)

    fig, ax1 = plt.subplots()
    if profile != 4:
        plt.plot(Egrid, z, label=PlotLabel)
    else:
        plt.plot(Egrid, z1, label="Lorentzian")
        plt.plot(Egrid, z2, label="Gaussian")
        plt.plot(Egrid, z3, label="Voigt")

# No sticks if overlapped plots
    if (nsets > 1) and (PlotType == 1):
        WithSticks = 0

    if WithSticks:
        plt.vlines(data[:, 0], [0], ScaleStick*data[:, 1], colors='black')

    plt.legend()
    plt.xlabel(XLabel)
    plt.ylabel(YLabel)
    plt.title(PlotTitle)
    ax1.xaxis.set_tick_params(direction='in', top='on', which='both')
    ax1.yaxis.set_tick_params(direction='in', top='on', which='both')
    plt.show()

    if SaveFile:
        if SaveFileType == 1:
            plt.savefig(SaveFileName+'.svg')
        elif SaveFileType == 2:
            plt.savefig(SaveFileName+'.pdf')
        elif SaveFileType == 3:
            plt.savefig(SaveFileName+'.png')
        else:
            print("Wrong SaveFileType!")
            sys.exit(1)

def ConvPlot_UVvis(gamma, PlotTitle, PlotLabel, profile=1, normalized=0,
                   EShift=0.0, nsets=1,
                   PlotType=1, WithSticks=1, EnergyMargin=0.1,
                   ExptDataFile="", DataFileName="simu_data.dat", NoPoints=1000,
                   XLabel="Energy (nm)", YLabel="Oscillation Strength (a. u.)",
                   SaveFile=False, SaveFileType=2, SaveFileName="plot", ScaleStick=1.0):

    # Read the data file: energy vs. oscillation strength
    data = genfromtxt(DataFileName, dtype='float')

    dim = len(data[:, 0])
    Emin = amin(data[:, 0])
    Emax = amax(data[:, 0])

# Build a 1D grid for plotting
    Estep = (Emax - Emin + 2.0*EnergyMargin)/NoPoints
    Egrid = arange(Emin-EnergyMargin, Emax+EnergyMargin, Estep)

# Creat a list of zeros
    z = zeros(len(Egrid))
    if profile == 4:
        z1 = zeros(len(Egrid))
        z2 = zeros(len(Egrid))
        z3 = zeros(len(Egrid))

# Sum on the grid
    for i in range(dim):
        if profile == 1:
            z = z + data[i, 1]*1.0/pi*gamma/2.0 / \
                ((Egrid-data[i, 0])**2+gamma*gamma/4.0)
        elif profile == 2:
            sigma = gamma/(2.0*sqrt(2.0*log(2.0)))
            z = z + data[i, 1]*exp(-0.5*((Egrid-data[i, 0])/sigma)
                               ** 2)/(sigma*sqrt(2.0*pi))
        elif profile == 3:
            zz = (Egrid-data[i, 0]+1j*gamma/2.0)/(sigma*sqrt(2.0))
#		wz = exp(-1.0*zz*zz)*special.erfc(-1j*zz)
# wofz: the SciPy implementation of the Faddeeva function
            z = z + data[i, 1]*special.wofz(zz).real/(sigma*sqrt(2.0*pi))
        elif profile == 4:
            z1 = z1 + data[i, 1]*1.0/pi*gamma/2.0 / \
                ((Egrid-data[i, 0])**2+gamma*gamma/4.0)
            z2 = z2 + \
                data[i, 1]*exp(-0.5*((Egrid-data[i, 0])/sigma)
                               ** 2)/(sigma*sqrt(2.0*pi))
            zz = (Egrid-data[i, 0]+1j*gamma/2.0)/(sigma*sqrt(2.0))
#                wz = exp(-1.0*zz*zz)*special.erfc(-1j*zz)
            z3 = z3 + data[i, 1]*special.wofz(zz).real/(sigma*sqrt(2.0*pi))
        else:
            print("Wrong lineshape option!")
            sys.exit(1)

    fig, ax1 = plt.subplots()
    if profile != 4:
        plt.plot(Egrid, z, label=PlotLabel)
    else:
        plt.plot(Egrid, z1, label="Lorentzian")
        plt.plot(Egrid, z2, label="Gaussian")
        plt.plot(Egrid, z3, label="Voigt")

# No sticks if overlapped plots
    if (nsets > 1) and (PlotType == 1):
        WithSticks = 0

    if WithSticks:
        plt.vlines(data[:, 0], [0], ScaleStick*data[:, 1], colors='black')

    plt.legend()
    plt.xlabel(XLabel)
    plt.ylabel(YLabel)
    plt.title(PlotTitle)
    ax1.xaxis.set_tick_params(direction='in', top='on', which='both')
    ax1.yaxis.set_tick_params(direction='in', top='on', which='both')
    plt.show()

    if SaveFile:
        if SaveFileType == 1:
            plt.savefig(SaveFileName+'.svg')
        elif SaveFileType == 2:
            plt.savefig(SaveFileName+'.pdf')
        elif SaveFileType == 3:
            plt.savefig(SaveFileName+'.png')
        else:
            print("Wrong SaveFileType!")
            sys.exit(1)

##################################################################################################
# General plot function for multiple simulation data sets with expt. data.
# SpecType: number; type of spectra; 1: XAS; 2: UV-vis; 3: Vibrational (IR and Raman)
# gamma: list of broadening factors (numbers) for multiple curves; len = no. of simulation curves
#        = l0
# PlotTitle: string; title of the plot
# PlotLabel: list of strings; len = l0
# profile: number; type of convolution function; 1: Lorentzian; 2: Gaussian; 3: Vogit; 
#          4: 1-3 combined, diagonistics only
# normalized: number; 0: max signal not normalized to 1; 1 or other: normalized
# Eshift: list of numbers; how much the energies are shifted for each curve; default to 
#         be all zeros; nonzero only for XAS
# EnergyMargin: number; margin for plotting the energy range
# ExptDataFile: the name of the file containing all raw expt. data (no numerical manipulation 
#               needed); default to be null
# DataFileNames: list of the names of the simulation data files; 
# NoPoints: number; no. of plotting points
# XLabel: string; label of the x-axis
# YLabel: string; label of the y-axis
# SaveFile: string; True: save plot to file; False: no save
# SaveFileType: number; 1: .svg; 2: .pdf; 3: .png
# SaveFileName: string; name of the saved figure file;
# ScaleStick: number; scaling factor to adjust the stick heights
# ScaleCurves: list of numbers; scalilng factors to adjust the curve heights
# VShifts: list of numbers; vertical shifts applied to multiple curves
##################################################################################################

def ConvPlot(SpecType, gamma, PlotTitle, PlotLabel=[""], profile=1, normalized=0,
                 EShift=[0.0], EnergyMargin=100.0,
                 ExptDataFile="", DataFileNames=["simu_data.dat"], NoPoints=1000,
                 XLabel="", YLabel="Oscillation Strength (arbitary unit)",
                 SaveFile=False, SaveFileType=2, SaveFileName="plot", ScaleStick=1.0,
                 ScaleExpt = 1.0, ScaleCurves=[1.0], VShifts=[0.0]):

    # how many curves we want to plot?
    l0 = len(DataFileNames)
    
    # sanity checks
    if l0 < 1:
        print("No plotting data!")
        sys.exit(1)
        
    if SpecType != 1:
        EShift = []
        for i in range(l0):
            EShift.append(0.0)

    if XLabel == "":
        if SpecType == 1:
            XLabel = "Energy (eV)"
        elif SpecType == 2:
            XLabel = "Energy (nm)"
        elif SpecType == 3:
            XLabel="Energy (cm^-1)"
        else:
            print("Wrong SpecType!")
            sys.exit(1)
        
    if (len(EShift) != l0) or (len(ScaleCurves) != l0):
        print("Curve parameter dimensions not equal to no. of curves!")
        sys.exit(1)
    
    if ExptDataFile != "":
        if  len(VShifts) != (l0 + 1):
            print("Curve parameter dimensions not equal to no. of curves!")
            sys.exit(1) 
    else:
        if  len(VShifts) != l0:
            print("Curve parameter dimensions not equal to no. of curves!")
            sys.exit(1)  
                      
    # Read the data files: energy vs. oscillation strength
    data = []
    
    for i in range(l0):
        data.append(genfromtxt(DataFileNames[i], dtype='float'))
    
    fig, ax1 = plt.subplots()    
    for j in range(l0):
        dim = len(data[j][:, 0])
        Emin = amin(data[j][:, 0])
        Emax = amax(data[j][:, 0])
    
    # Build a 1D grid for plotting
        Estep = (Emax - Emin + 2.0*EnergyMargin)/NoPoints
        Egrid = arange(Emin-EnergyMargin, Emax+EnergyMargin, Estep)
    
    # Creat a list of zeros
        z = zeros(len(Egrid))
        if profile == 4:
            z1 = zeros(len(Egrid))
            z2 = zeros(len(Egrid))
            z3 = zeros(len(Egrid))
    
    # Sum on the grid
        for i in range(dim):
            if profile == 1:
                z = z + data[j][i, 1]*1.0/pi*gamma/2.0 / \
                    ((Egrid-data[j][i, 0])**2+gamma*gamma/4.0)
            elif profile == 2:
                sigma = gamma/(2.0*sqrt(2.0*log(2.0)))
                z = z + data[j][i, 1]*exp(-0.5*((Egrid-data[j][i, 0])/sigma)
                                   ** 2)/(sigma*sqrt(2.0*pi))
            elif profile == 3:
                zz = (Egrid-data[j][i, 0]+1j*gamma/2.0)/(sigma*sqrt(2.0))
    #		wz = exp(-1.0*zz*zz)*special.erfc(-1j*zz)
    # wofz: the SciPy implementation of the Faddeeva function
                z = z + data[j][i, 1]*special.wofz(zz).real/(sigma*sqrt(2.0*pi))
            elif profile == 4:
                z1 = z1 + data[j][i, 1]*1.0/pi*gamma/2.0 / \
                    ((Egrid-data[j][i, 0])**2+gamma*gamma/4.0)
                z2 = z2 + \
                    data[j][i, 1]*exp(-0.5*((Egrid-data[j][i, 0])/sigma)
                                   ** 2)/(sigma*sqrt(2.0*pi))
                zz = (Egrid-data[j][i, 0]+1j*gamma/2.0)/(sigma*sqrt(2.0))
    #                wz = exp(-1.0*zz*zz)*special.erfc(-1j*zz)
                z3 = z3 + data[j][i, 1]*special.wofz(zz).real/(sigma*sqrt(2.0*pi))
            else:
                print("Wrong lineshape option!")
                sys.exit(1)
    
        if profile != 4:
            plt.plot(Egrid, ScaleCurves[j] * z + VShifts[j], label=PlotLabel[j])
        else:
            plt.plot(Egrid, ScaleCurves[j] * z1 + VShifts[j], label="Lorentzian_"+str(j+1))
            plt.plot(Egrid, ScaleCurves[j] * z2 + VShifts[j], label="Gaussian_"+str(j+1))
            plt.plot(Egrid, ScaleCurves[j] * z3 + VShifts[j], label="Voigt_"+str(j+1))
    
    if ExptDataFile != "":
        data_expt = genfromtxt(ExptDataFile, dtype='float')
        dim_expt = len(data_expt[:, 0])
        plt.plot(data_expt[:, 0], ScaleExpt * data_expt[:, 1] + VShifts[-1], label="Expt.")

# No sticks for multiple curve plots
    if l0 == 1:
        plt.vlines(data[:, 0], [0], ScaleStick*data[:, 1], colors='black')

    plt.legend()
    plt.xlabel(XLabel)
    plt.ylabel(YLabel)
    plt.title(PlotTitle)
    ax1.xaxis.set_tick_params(direction='in', top='on', which='both')
    ax1.yaxis.set_tick_params(direction='in', top='on', which='both')
    plt.show()

    if SaveFile:
        if SaveFileType == 1:
            plt.savefig(SaveFileName+'.svg')
        elif SaveFileType == 2:
            plt.savefig(SaveFileName+'.pdf')
        elif SaveFileType == 3:
            plt.savefig(SaveFileName+'.png')
        else:
            print("Wrong SaveFileType!")
            sys.exit(1)

# Plotting vibration spectroscopy signals (IR, Raman)
def ConvPlot_Vib(gamma, PlotTitle, PlotLabel, profile=1, normalized=0,
                 EShift=0.0, nsets=1, PlotType=1, WithSticks=1, EnergyMargin=100.0,
                 ExptDataFile="", DataFileName="simu_data.dat", NoPoints=1000,
                 XLabel="Energy (cm^-1)", YLabel="Oscillation Strength (a. u.)",
                 SaveFile=False, SaveFileType=2, SaveFileName="plot", ScaleStick=1.0,
                 ScaleExpt=1.0):

    # Read the data file: energy vs. oscillation strength
    data = genfromtxt(DataFileName, dtype='float')

    dim = len(data[:, 0])
    Emin = amin(data[:, 0])
    Emax = amax(data[:, 0])

# Build a 1D grid for plotting
    Estep = (Emax - Emin + 2.0*EnergyMargin)/NoPoints
    Egrid = arange(Emin-EnergyMargin, Emax+EnergyMargin, Estep)

# Creat a list of zeros
    z = zeros(len(Egrid))
    if profile == 4:
        z1 = zeros(len(Egrid))
        z2 = zeros(len(Egrid))
        z3 = zeros(len(Egrid))

# Sum on the grid
    for i in range(dim):
        if profile == 1:
            z = z + data[i, 1]*1.0/pi*gamma/2.0 / \
                ((Egrid-data[i, 0])**2+gamma*gamma/4.0)
        elif profile == 2:
            sigma = gamma/(2.0*sqrt(2.0*log(2.0)))
            z = z + data[i, 1]*exp(-0.5*((Egrid-data[i, 0])/sigma)
                               ** 2)/(sigma*sqrt(2.0*pi))
        elif profile == 3:
            zz = (Egrid-data[i, 0]+1j*gamma/2.0)/(sigma*sqrt(2.0))
#		wz = exp(-1.0*zz*zz)*special.erfc(-1j*zz)
# wofz: the SciPy implementation of the Faddeeva function
            z = z + data[i, 1]*special.wofz(zz).real/(sigma*sqrt(2.0*pi))
        elif profile == 4:
            z1 = z1 + data[i, 1]*1.0/pi*gamma/2.0 / \
                ((Egrid-data[i, 0])**2+gamma*gamma/4.0)
            z2 = z2 + \
                data[i, 1]*exp(-0.5*((Egrid-data[i, 0])/sigma)
                               ** 2)/(sigma*sqrt(2.0*pi))
            zz = (Egrid-data[i, 0]+1j*gamma/2.0)/(sigma*sqrt(2.0))
#                wz = exp(-1.0*zz*zz)*special.erfc(-1j*zz)
            z3 = z3 + data[i, 1]*special.wofz(zz).real/(sigma*sqrt(2.0*pi))
        else:
            print("Wrong lineshape option!")
            sys.exit(1)

    fig, ax1 = plt.subplots()
    if profile != 4:
        plt.plot(Egrid, z, label=PlotLabel)
    else:
        plt.plot(Egrid, z1, label="Lorentzian")
        plt.plot(Egrid, z2, label="Gaussian")
        plt.plot(Egrid, z3, label="Voigt")

    if ExptDataFile != "":
        data_expt = genfromtxt(ExptDataFile, dtype='float')
        dim_expt = len(data_expt[:, 0])
        plt.plot(data_expt[:, 0], ScaleExpt * data_expt[:, 1], label="Expt.")

# No sticks if overlapped plots
    if (nsets > 1) and (PlotType == 1):
        WithSticks = 0

    if WithSticks:
        plt.vlines(data[:, 0], [0], ScaleStick*data[:, 1], colors='black')

    plt.legend()
    plt.xlabel(XLabel)
    plt.ylabel(YLabel)
    plt.title(PlotTitle)
    ax1.xaxis.set_tick_params(direction='in', top='on', which='both')
    ax1.yaxis.set_tick_params(direction='in', top='on', which='both')
    plt.show()

    if SaveFile:
        if SaveFileType == 1:
            plt.savefig(SaveFileName+'.svg')
        elif SaveFileType == 2:
            plt.savefig(SaveFileName+'.pdf')
        elif SaveFileType == 3:
            plt.savefig(SaveFileName+'.png')
        else:
            print("Wrong SaveFileType!")
            sys.exit(1)
