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

###########################################################
# Input Section
# Choose the convolution function: 1 = Lorentzian
# 2 = Gaussian
# 3 = Vogit
# 4 = 1-3 combined, diagnostics only
##
#profile = 1
# Full width at half maximum: C1s 0.06 eV
# N1s 0.09
# O1s 0.13
# Ref.: Handbook of X-ray Data, Zschornack, Springer 2007
# gamma = 0.13 # For Lorentzian
# sigma = gamma/(2.0*sqrt(2.0*log(2.0))) # For Gaussian
# Normalized or not? 0 = not normalized, 1 or other = normalized
#normalized = 0
# If normalized, choose type: 1 = area normalized
# 2 = max peak height normalized
# Energy shifted? Useful for X-ray spectroscopy simulation
#EShift = 0.0
# No. of data sets
#nsets = 1
# Plot type: 1 = overlap; 2 = stacked
#PlotType = 1
# With sticks?
#WithSticks = 1
# Energy unit: 1 = eV; 2 = nm; 3 = cm-1
#EnergyUnit = 1
# Energy margin for plotting
#EnergyMargin = 2.0
# Expt. data file name
# ExptDataFile =  "" # if not null then plot expt. data
# Simulation Data file name
##SimuDataFileNames = ["data-1.dat"]
#DataFileName = "tyr-O1s-23-td.dat"
# No. of plotting points
#NoPoints = 1000
# Plot labels and title
#PlotTitle = "XANES Plot Test"
#XLabel = "Energy (eV)"
#YLabel = "Oscillation Strength (a. u.)"
# Save file type and name
# SaveFileType = 2 # 1: .svg; 2: .pdf; 3: .png
#SaveFileName = "plot"
# End of Input Section
###########################################################


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


def ConvPlot_Vib(gamma, PlotTitle, PlotLabel, profile=1, normalized=0,
                 EShift=0.0, nsets=1, PlotType=1, WithSticks=1, EnergyMargin=100.0,
                 ExptDataFile="", DataFileName="simu_data.dat", NoPoints=1000,
                 XLabel="Energy (cm^-1)", YLabel="Oscillation Strength (a. u.)",
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
