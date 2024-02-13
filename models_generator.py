# Import necessary libraries
import sys
import json
import datetime
import numpy as np
from xspec import AllModels, AllData, Model, Plot
from pathlib import Path

# Import custom modules
from modules.utils import (
    extract_number,
    read_last_successful_index,
    save_last_successful_index,
    )
from modules.logging_config import logging_conf

if __name__ == "__main__":
     
    if len(sys.argv) < 2:
        print("Usage: python my_script.py <filename>")
        sys.exit(1)
    
    sample_file_name = sys.argv[1]

    # Set smoothing
    smoothing = True
    # Set checkpoint file names
    extracted_number = extract_number(sample_file_name)
    index_file_name = f'last_successful_index_{extracted_number}.txt'

    # Set up paths for logs and models
    cwd = Path.cwd()
    path_to_logs = Path(cwd / "logs")
    path_to_logs.mkdir(parents=True, exist_ok=True)
    path_to_models = Path(cwd / "models")
    path_to_models.mkdir(parents=True, exist_ok=True)
    path_to_samples = Path(cwd / "samples")
    path_to_checkpoints = Path(cwd / "checkpoints")
    path_to_checkpoints.mkdir(parents=True, exist_ok=True)
    sample_file_path = path_to_samples / sample_file_name
    index_file_path = path_to_checkpoints / index_file_name

    # Get the current date and time
    t_start = datetime.datetime.now()
    # Format the current date and time as a string
    timestamp = t_start.strftime("%Y-%m-%d_%H-%M-%S")

    # Set up log configuration and create a logger for the fit
    logger = logging_conf(path_to_logs, f"models_creator_{extracted_number}_{timestamp}.log")

    # Debug: Log the start of the script
    logger.debug("Script started.")

    model_name = "TBabs*(rdblur*rfxconv*comptb + diskbb + comptb)"

    logger.debug("Latin Hypercube exists already. Loading from disk.")
    sample_scaled = np.load(sample_file_path)
    relevant_par =  np.load(path_to_samples / "relevant_par.npy")

    # Invert the log10 of these components
    log_index = [0, 9, 12]
    for i in range(sample_scaled.shape[0]):
        for j in log_index:
            sample_scaled[i, j] =  pow(10, sample_scaled[i, j])

    # Check if index_file exists
    if (index_file_path).is_file():
        last_successful_index = read_last_successful_index(index_file_path)
    else:
        last_successful_index = 0
        
    # Iterate through scaled samples to set up and save models
    for index, params in enumerate(sample_scaled):

        idx = index + extracted_number
        
        if idx < last_successful_index:
            continue  # Skip already processed samples
        try:
            # Debug: Log parameters for the current iteration
            logger.debug(f"Step number: {idx}/{len(sample_scaled) + extracted_number}")
            logger.debug(f"Current parameters: {params}")

            # Clear existing XSPEC models and data
            AllModels.clear()
            AllData.clear()
            AllData.dummyrsp(0.5, 20.)

            # Initialize the model
            m = Model(model_name)

            # Changing default frozen parameters to unfrozen
            m.rdblur.Betor10.frozen = False
            m.rdblur.Rout_M.frozen = True
            m.rdblur.Rin_M.frozen = False
            m.rfxconv.Fe_abund.frozen = False
            m.comptb.gamma.frozen = True
            m.comptb.delta.frozen = True
            m.comptb.log_A.frozen = True

            m.rdblur.Rout_M.values = 1000
            m.comptb.delta.values = 0
            m.comptb.log_A.values = 8

            m.rfxconv.cosIncl.link = "COSD(5)"
            # Linking comptb_6 (refletion) parameters to comptb (comptb)
            start = 20  # Number of the first parameter of comptb_6
            for i in range(start, start + len(m.comptb_6.parameterNames)):
                m(i).link = m(i-9) # 9 is the separation between comptb and comptb_6

            # Add the model to the spectral analysis system and set parameters
            AllModels.setPars(m, {int(relevant_par[j]):params[j] for j in range(len(relevant_par))})

            # Set up the energy range of interest for plotting
            Plot.device = "/null"
            Plot.xAxis = "keV"
            Plot.show()
            Plot('model')
            energy = Plot.x()
            flux = Plot.model()

            # Smooth the data by averaging every 'n_points' consecutive points
            if smoothing:
                energy = np.array(energy)
                flux = np.array(flux)
                n_points = 10
                energy = (energy.reshape(-1, n_points).mean(axis=1)).tolist()
                flux = (flux.reshape(-1, n_points).mean(axis=1)).tolist()

            # Create a dictionary for the parameters
            params_dict = {}
            for i in range(1, m.nParameters+1):
                if not m(i).frozen and not m(i).link:
                    # Restore the log10 value and save it in the json
                    if i in [1, 15, 19]:
                        params_dict[str(i)+" log"] = np.log10(m(i).values[0])
                    else:
                        params_dict[str(i)] = m(i).values[0]
            
            # Store parameters and data in a dictionary
            data = {
                'parameters' : params_dict,
                'energy (keV)': energy,
                'flux (1 / keV cm^-2 s)': flux
            }

            if len(params) < 6:
                # Create the file name based on parameter values
                file_name = f"model_{idx:05d}_params" + "_".join([f"{format(param, '.1e')}" for param in params])
            else:
                file_name = f"model_{idx:05d}"

            # Save the dictionary as json with the created file name
            with open( path_to_models / f'{file_name}.json', 'w') as json_file:
                json.dump(data, json_file)
        
            # Save the current index as the last successful one
            save_last_successful_index(idx, index_file_path)
        
        except Exception as e:
            logger.error(f"An error occurred at index {idx}: {e}")
            break  # or handle the error as needed

    # Get the current date and time
    t_stop = datetime.datetime.now()
    logger.debug(f"execution time: {t_stop - t_start}")
    # Debug: Log the end of the script
    logger.debug("Script completed.")
