import os
from adcircpy import Tides
from adcircpy.server import SlurmConfig
from adcircpy import AdcircMesh, AdcircRun

def main():
    # Paths
    mesh_file = "fort.14"
    tpxo_path = r"C:\TPXO_10_atlas_V2\h_tpxo10_v2.nc"  # Pointing to the specific h_tpxo10 file, adjust if necessary
    
    print(f"Loading mesh from {mesh_file}...")
    mesh = AdcircMesh.open(mesh_file, crs="EPSG:4326")
    
    # Alternatively, pointing to the folder if adcircpy supports it
    print(f"Applying TPXO tides from {r'C:\\TPXO_10_atlas_V2'}...")
    # Initialize the Tides forcing
    tides = Tides()
    tides.use_tpxo(r"C:\TPXO_10_atlas_V2") # Typically adcircpy expects the path to the directory
    
    # Note: If adcircpy requires building a full run object to write fort.15:
    # run = AdcircRun(mesh=mesh, ...)
    # run.add_forcing(tides)
    # run.write('path')
    # Because we already have a fort.15, we might need to parse and inject it, 
    # but adcircpy usually rewrites the whole fort.15.
    
    print("Script template created. Please integrate with your AdcircRun setup to write the updated fort.15.")

if __name__ == "__main__":
    main()
