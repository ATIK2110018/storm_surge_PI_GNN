import os
import subprocess
import urllib.request

def main():
    work_dir = r"C:\Users\atikr\OceanMesh2D\actual_amphan"
    os.chdir(work_dir)
    print("=== Phase 1: Generating Mesh & Control Files via MATLAB ===")

    # 1. Write the precise MATLAB configuration script
    matlab_script = """
addpath('C:\\Users\\atikr\\OceanMesh2D');
addpath('C:\\Users\\atikr\\OceanMesh2D\\utilities');
disp('Loading fort.14...');
m = msh('fort.14');
disp('Configuring Make_f15 with correct WTIMNC array for ATCF NWS=8...');
% Simulation runs for 3 days matching the downloaded ATCF wind track (16-May to 19-May)
m = Make_f15(m, '16-May-2020 00:00', '19-May-2020 09:00', 2.0, 'NWS', 8, ...
    'WTIMNC', [2020 5 16 0 1 1.0], 'RSTIMNC', 3, 'const', {'M2', 'S2', 'N2', 'K1', 'O1'}, ...
    'tidal_database', 'C:\\TPXO_10_atlas_V2\\TPXO10_atlas_v2_nc\\h_**_tpxo10_atlas_30_v2.nc', ...
    'namelist', {'swanoutput', 'nws8'});
disp('Writing fort.15...');
write(m, 'fort.15', 'f15');
disp('Done MATLAB generation.');
exit;
"""
    with open("generate_model.m", "w") as f:
        f.write(matlab_script)
    
    # Execute MATLAB
    subprocess.run(["matlab", "-batch", "generate_model"], check=True)
    
    # Make_f15 writes to fort.15.15 if fort.15 exists, rename it if so
    if os.path.exists("fort.15.15"):
        if os.path.exists("fort.15"):
            os.remove("fort.15")
        os.rename("fort.15.15", "fort.15")

    print("=== Phase 2: Patching fort.15 (Removing metadata) ===")
    
    # 2. Fix fort.15 (Strip trailing OceanMesh2D metadata)
    with open("fort.15", "r") as f:
        content = f.read()
    
    # Find the start of the OceanMesh2D metadata block
    idx = content.find("OceanMesh2D \nAffiliation")
    if idx != -1:
        content = content[:idx]
        with open("fort.15", "w", newline="\n") as f: # Force LF endings
            f.write(content)
        print("Successfully stripped OceanMesh2D metadata from fort.15.")

    print("=== Phase 3: Downloading ATCF best track (fort.22) ===")

    # 3. Download the actual Cyclone Amphan ATCF track (IO012020)
    url = "https://www.metoc.navy.mil/jtwc/products/best-tracks/2020/2020s-bio/bio012020.txt"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    
    try:
        with urllib.request.urlopen(req) as response:
            atcf_data = response.read().decode('utf-8')
        with open("fort.22", "w", newline="\n") as f:
            f.write(atcf_data)
        print("Downloaded official ATCF track to fort.22.")
    except Exception as e:
        print(f"Failed to download ATCF: {e}")

    print("=== Phase 4: Patching SWAN files (fort.26 & swaninit) ===")

    # 4. Clean up SWAN configuration files
    # Remove COMPUTE and STOP from fort.26 because ADCIRC handles the time loop
    if os.path.exists("fort.26"):
        with open("fort.26", "r") as f:
            f26_lines = f.readlines()
        
        f26_cleaned = [line for line in f26_lines if "COMPUTE" not in line and "STOP" not in line]
        
        with open("fort.26", "w", newline="\n") as f:
            f.writelines(f26_cleaned)
        print("Patched fort.26 (Removed COMPUTE/STOP).")

    # Rewrite swaninit to ensure exact spacing and no \r (Carriage Return)
    swaninit_content = "    4                                   version of swaninit file\n    0                                   delc\n                                        print\n                                        fort.26\n"
    with open("swaninit", "w", newline="\n") as f:
        f.write(swaninit_content)
    print("Patched swaninit.")

    print("=== Phase 5: Generating run script ===")

    # 5. Generate bash run script for WSL
    run_script = """#!/bin/bash
ADCDIR=/home/atikr/asgs/opt/models/adcircs/adcirc-cg-v56.0.4.live.0-gfortran-wsl-ubuntu/work
export PATH=$PATH:$ADCDIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/atikr/asgs/opt/lib:/home/atikr/miniconda3/envs/adcirc_tools/lib

echo "Launching adcirc on a single core..."
adcirc

echo "ADCIRC Run Completed!"
"""
    with open("run_model.sh", "w", newline="\n") as f:
        f.write(run_script)
    
    # Make executable in WSL just in case (optional, we use bash ./run_model.sh anyway)
    print("Generated run_model.sh.")
    print("\n✅ Preparation Complete! You can now run `./run_model.sh` in WSL.")

if __name__ == "__main__":
    main()
