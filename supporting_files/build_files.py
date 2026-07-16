import os
import urllib.request

# Directory
out_dir = r"C:\Users\atikr\OceanMesh2D\actual_amphan"

# 1. fort.15
fort15_path = os.path.join(out_dir, "fort.15")
with open(fort15_path, "w") as f:
    f.write("""Amphan Wind-Only Storm Surge Test
 0151 ! RUNID
 X ! NFOVER
 0 ! NABOUT
 -1 ! NSCREEN
 0 ! IHOT (Cold Start)
 30 ! ICS (Spherical coordinates)
 1 ! IM (Fully Implicit/Semi-Implicit)
 8 ! NWS (Holland Wind Model from fort.22)
 1 ! NRAMP
 0.380 ! G (Gravity)
 1.0 ! TAU0
 2.0 ! DT (Time step in seconds)
 1.0 ! STATIM (Start time in days)
 1.0 ! REFTIM
 8.0 ! WTIMINC (Wind time increment)
 8.0 ! RUNDES (Run duration in days)
 0.0 ! Z0 (Initial water level)
 0 ! NBFR (Number of tidal constituents)
 0 ! ANGINN
 0 ! NOUTGE (Global Elevation Output)
 1 ! TOUTGE
 1 ! TOUTGV
 0 ! NOUTGW (Global Velocity Output)
""")

# 2. fort.22
fort22_path = os.path.join(out_dir, "fort.22")
url = "https://www.metoc.navy.mil/jtwc/products/best-tracks/2020/2020s-bio/bio012020.txt"
try:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    response = urllib.request.urlopen(req)
    data = response.read().decode('utf-8')
    lines = [line.strip() for line in data.splitlines() if line.strip()]
    
    with open(fort22_path, "w") as f:
        f.write("AMPHAN\n")
        f.write(f" {len(lines)}\n")
        
        for row in lines:
            parts = [x.strip() for x in row.split(',')]
            dt = parts[2]
            
            lat_str = parts[6]
            lat = float(lat_str[:-1]) / 10.0
            if lat_str.endswith('S'): lat = -lat
                
            lon_str = parts[7]
            lon = float(lon_str[:-1]) / 10.0
            if lon_str.endswith('W'): lon = -lon
                
            vmax = float(parts[8])
            pmin = float(parts[9])
            
            rmax = 30.0
            if len(parts) >= 20:
                try:
                    parsed_rmax = float(parts[19])
                    if parsed_rmax > 0:
                        rmax = parsed_rmax
                except ValueError:
                    pass
            
            f.write(f" {dt}   {lat:.2f}   {lon:.2f}     {vmax:.1f}    {pmin:.1f}    {rmax:.1f}\n")
    print("fort.22 generated successfully.")
except Exception as e:
    print(f"Error downloading or parsing JTWC data: {e}")

# 3. fort.26 (SWAN control file)
fort26_path = os.path.join(out_dir, "fort.26")
with open(fort26_path, "w") as f:
    f.write("""$ SWAN control file for ADCIRC+SWAN coupled run
PROJ 'Amphan' '01'
$
$ ADCIRC passes Wind to SWAN and Water Levels
COORDINATES SPHERICAL
$
$ Define spectral space
CGRID UNSTRUCTURED CIRCLE 36 0.04 1.0 30
$
$ Physical processes
WIND
GEN3 KOMEN
QUAD
WCAP
FRIC JONSWAP
BREAKING
$
$ Numerics
PROP BSBT
$
NUMERIC COMPACT 1e-4 100
$
$ Output commands for block output or point output if needed
$ BLOCK 'COMPGRID' NOHEADER 'swan_Hs.mat' LAY 3 HSIGN 1.0
$
TEST 0, 0
COMPUTE
STOP
""")

# 4. swaninit
swaninit_path = os.path.join(out_dir, "swaninit")
with open(swaninit_path, "w") as f:
    f.write("""    4                                   version of swaninit file
    0    0                              delc, ismax
                                        noprint
                                        adcirc
""")

print("All files generated successfully.")
