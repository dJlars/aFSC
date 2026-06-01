import os
import shutil
import configargparse

def get_config_Duffingx3(config_file, result_folder = None, logger = None):

    p = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    p.add('-c', default = config_file, is_config_file=True, help='config file path')
    p.add('--alpha', default = 7, type=float, help='linear stiffness')
    p.add('--beta', default = 5, type=float, help='nonlinear stiffness')
    p.add('--delta', default = 0.01, type=float, help='damping')
    p.add('--gamma', default = 0.2, type=float, help='force amplitude')
    p.add('--omega', default = 3, type=float, help='force frequency')
    p.add('--nrEvalPts', default = 1000, type=int, help='number of evaluation points')
    p.add('--x0', default = 0, type=float, help='Start time')
    p.add('--v0', default = 14, type=float, help='Start time')
    p.add('--t0', default = 0, type=float, help='Start time')
    p.add('--t_end_tr', default = 300, type=int, help='Transient process end time in minutes')
    p.add('--t_step_tr', default = 1000, type=float, help='Transient process time step in miliseconds')
    p.add('--t_end_ss', default = 100, type=int, help='End time in minutes')
    p.add('--t_step_ss', default = 500, type=float, help='Time step in miliseconds')
    p.add('--fftMinVal', default = 0.001, type=float, help='FFT Threshold value')
    p.add('--fftHighFreq', default = 500, type=float, help='FFT highest frequency')
    p.add('--fftDist', default = 50, type=float, help='FFT distance between peaks')
    
    
    config = p.parse_args()


    # write configs to logger
    if logger is not None: logger.info("User input (cmd > config > default): \n" + p.format_values())    

    # Copy config file to result path
    if result_folder is not None:
        config_filename = os.path.basename(config_file)
        shutil.copy(config_file, result_folder + "/" + config_filename)

    return config

def get_config_Duffingx3(config_file, result_folder = None, logger = None):

    p = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    p.add('-c', default = config_file, is_config_file=True, help='config file path')
    p.add('--alpha', default = 7, type=float, help='linear stiffness')
    p.add('--beta', default = 5, type=float, help='nonlinear stiffness')
    p.add('--delta', default = 0.01, type=float, help='damping')
    p.add('--gamma', default = 0.2, type=float, help='force amplitude')
    p.add('--omega', default = 3, type=float, help='force frequency')
    p.add('--nrEvalPts', default = 1000, type=int, help='number of evaluation points')
    p.add('--x0', default = 0, type=float, help='Start time')
    p.add('--v0', default = 14, type=float, help='Start time')
    p.add('--t0', default = 0, type=float, help='Start time')
    p.add('--t_end_tr', default = 300, type=int, help='Transient process end time in minutes')
    p.add('--t_step_tr', default = 1000, type=float, help='Transient process time step in miliseconds')
    p.add('--t_end_ss', default = 100, type=int, help='End time in minutes')
    p.add('--t_step_ss', default = 500, type=float, help='Time step in miliseconds')
    p.add('--fftMinVal', default = 0.001, type=float, help='FFT Threshold value')
    p.add('--fftHighFreq', default = 500, type=float, help='FFT highest frequency')
    p.add('--fftDist', default = 50, type=float, help='FFT distance between peaks')
    
    
    config = p.parse_args()


    # write configs to logger
    if logger is not None: logger.info("User input (cmd > config > default): \n" + p.format_values())    

    # Copy config file to result path
    if result_folder is not None:
        config_filename = os.path.basename(config_file)
        shutil.copy(config_file, result_folder + "/" + config_filename)

    return config

def get_config_Cell(config_file, result_folder, logger = None):

    p = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    p.add('-c', default = config_file, is_config_file=True, help='config file path')
    p.add('--V_0', default = -60, type=float, help='Initial value of V in µM')
    p.add('--n_0', default = 0, type=float, help='Initial value of n in µM')
    p.add('--Ca_0', default = 0.1, type=float, help='Initial value of Ca in µM')
    p.add('--t0', default = 0, type=float, help='Start time')
    p.add('--t_end_tr', default = 7, type=int, help='Transient process end time in minutes')
    p.add('--t_step_tr', default = 1000, type=float, help='Transient process time step in miliseconds')
    p.add('--t_end_ss', default = 24, type=int, help='End time in minutes')
    p.add('--t_step_ss', default = 500, type=float, help='Time step in miliseconds')
    p.add('--fftMinVal', default = 0.001, type=float, help='FFT Threshold value')
    p.add('--fftHighFreq', default = 500, type=float, help='FFT highest frequency')
    p.add('--fftDist', default = 50, type=float, help='FFT distance between peaks')
    p.add('--nrEvalPts', default = 1000, type=int, help='number of evaluation points')
    p.add('--atp', default = 1800, type=float, help='mean atp concentration')
    
    config = p.parse_args()


    # write configs to logger
    if logger is not None: logger.info("User input (cmd > config > default): \n" + p.format_values())    

    # Copy config file to result path
    config_filename = os.path.basename(config_file)
    shutil.copy(config_file, result_folder + "/" + config_filename)

    return config

def get_config_Beam(config_file, result_folder = None, logger = None):

    p = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    p.add('-c', default = config_file, is_config_file=True, help='config file path')
    p.add('--l', default = 0.58, type=float, help='beam length')
    p.add('--h', default = 0.002, type=float, help='beam height')
    p.add('--b', default = 0.02, type=float, help='beam width')
    p.add('--E', default = 700000000, type=float, help='elasticity')
    p.add('--rho', default = 2778, type=float, help='density')
    p.add('--gamma', default = 0.03, type=float, help='excitation amplitude')
    p.add('--omega', default = 80, type=float, help='excitation frequency')
    p.add('--nrEvalPts', default = 1000, type=int, help='number of evaluation points')
    
    config = p.parse_args()


    # write configs to logger
    if logger is not None: logger.info("User input (cmd > config > default): \n" + p.format_values())    

    # Copy config file to result path
    if result_folder is not None:
        config_filename = os.path.basename(config_file)
        shutil.copy(config_file, result_folder + "/" + config_filename)

    return config

def get_config_nvhSystem(config_file, result_folder = None, logger = None):

    p = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    # p.add('-c', default = config_file, is_config_file=True, help='config file path')
    p.add('--alpha', default = 1.6e-7, type=float, help='mass damp factor')
    p.add('--beta', default = 0.002, type=float, help='stiffness damp factor')
    p.add('--kl', default = 1.8e7, type=float, help='contact linear stiffness')
    p.add('--knl', default = 5e9, type=float, help='contact nonlinear stiffness')
    p.add('--F', default = 7000, type=float, help='pressure force')
    p.add('--mu', default = 0.2, type=float, help='friction coefficient')
    p.add('--padE', default = 2e8, type=float, help='pad, E-modulus')
    p.add('--padPosionRatio', default = 0.1, type=float, help='pad, posion ratio')
    p.add('--padDensity', default = 2500, type=float, help='pad, density')
    p.add('--discE', default = 125e8, type=float, help='disc, E-modulus')
    p.add('--discPosionRatio', default = 0.3, type=float, help='disc, posion ratio')
    p.add('--discDensity', default = 7200, type=float, help='disc, density')
    p.add('--nrEvalPts', default = 500, type=int, help='number of evaluation points')
    
    config = p.parse_args()


    # write configs to logger
    if logger is not None: logger.info("User input (cmd > config > default): \n" + p.format_values())    

    # Copy config file to result path
    if result_folder is not None:
        config_filename = os.path.basename(config_file)
        shutil.copy(config_file, result_folder + "/" + config_filename)

    return config