# ADCIRC + SWAN Coupled Model Troubleshooting & Setup Guide

This document summarizes the challenges faced while setting up and running the ADCIRC+SWAN coupled model for the Amphan event (May 18–22, 2020), along with their solutions. It serves as a reference to prevent these issues in future model setups.

## 1. SIGFPE (Floating Point Exception) at "Begin Timestepping"
* **Problem**: The `adcswan` executable consistently crashed immediately after printing "Begin timestepping," outputting a `SIGFPE` backtrace.
* **Root Cause**: The bash environment lacked the `LD_LIBRARY_PATH` required to load dynamically linked libraries (specifically `libnetcdff.so` and `netcdf.so`). Without these, any initialization of NetCDF output or dynamic Fortran modules resulted in an immediate crash.
* **Solution**: Explicitly export the library path in your run script (`run_model.sh`) before launching the executable:
  ```bash
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/atikr/asgs/opt/lib
  ```

## 2. SWAN Control File Ignored
* **Problem**: SWAN parameters in `fort.26` were being ignored, or the model was initializing with incorrect defaults/crashing.
* **Root Cause**: The `swaninit` initialization file was instructing SWAN to look for a control file named `INPUT` instead of `fort.26`.
* **Solution**: Edit the `swaninit` file and change line 4 to explicitly point to `fort.26`:
  ```text
      3                                   command file ref. number
  fort.26                                 command file name
      4                                   print file ref. number
  ```

## 3. Invalid Timestamps ("00 Jan 0000") and File Parsing Errors
* **Problem**: Output NetCDF files showed incorrect dates (e.g., 00 Jan 0000), or the model crashed abruptly while reading `fort.15`.
* **Root Cause**: When using coupled wind models like `NWS=308`, modern versions of ADCIRC require a very strict 7-parameter format on line 23 of the `fort.15` file. Failing to provide exactly 7 values breaks the Fortran parser.
* **Solution**: Ensure line 23 of `fort.15` strictly follows this format: `YYYY MM DD HH StormNumber BLAdj WTIMINC`. 
  * *Example for Amphan*: `2020 5 18 0 1 1 10800.0`

## 4. GWCE Solver and SWAN Grid Stability
* **Problem**: Unstructured grid implementations of SWAN coupled with ADCIRC can sometimes exhibit instability at very shallow nodes or with specific wave generation physics.
* **Solution / Best Practices Applied**:
  * **Minimum Depth Matching**: Ensure ADCIRC's `H0` (minimum depth in `fort.15`) strictly matches SWAN's minimum depth (`DEPMIN`) in `fort.26`. For this model, both were set to `0.1`.
  * **GWCE Weighting**: Adjusted `TAU0` to `0.02` in `fort.15` to ensure numerical stability of the Generalized Wave Continuity Equation (GWCE).
  * **SWAN Physics**: Switched SWAN wave physics to `GEN3 JANSSEN` in `fort.26` for robust wave-action calculations on unstructured grids.

## 5. Ensuring Proper NetCDF Outputs
* **Problem**: ADCIRC and SWAN were running but not producing the desired `.nc` output files.
* **Solution**:
  * **For ADCIRC (`fort.15`)**: Set the output frequency variables (`NOUTGE`, `NOUTE`, `NOUTV`, `NOUTGM`, etc.) to negative values (e.g., `-5`) which tells ADCIRC to write outputs in NetCDF format instead of ASCII.
  * **For SWAN (`fort.26`)**: Uncommented the `BLOCK` commands and appended `.nc` to the filenames to enforce NetCDF output.
    ```text
    BLOCK 'COMPGRID' NOHEADER 'swan_HS.63.nc' LAY 3 HSIGN 1.0
    BLOCK 'COMPGRID' NOHEADER 'swan_DIR.63.nc' LAY 3 DIR 1.0
    BLOCK 'COMPGRID' NOHEADER 'swan_TPS.63.nc' LAY 3 RTP 1.0
    BLOCK 'COMPGRID' NOHEADER 'swan_WIND.63.nc' LAY 3 WIND 1.0
    ```

## 6. Pre-run Cleanups
* **Problem**: Old corrupted or partial NetCDF files can cause ADCIRC to abort early if it attempts to append to them with conflicting timestamps.
* **Solution**: Included an aggressive cleanup command in `run_model.sh` before starting `adcswan`:
  ```bash
  rm -rf fort.16 fort.33 fort.61.nc fort.62.nc fort.63.nc fort.64.nc fort.73.nc fort.74.nc swan_*.nc PE0000 PE0001
  ```

## 7. Numerical Instability at Open Ocean Boundaries (The "Cyclone Crossing" Blowup)
* **Problem**: The model frequently aborted around 42-46% completion (approx. 40-44 hours simulated) with massive `ELMAX` (>1000m) and `SPEEDMAX` (>360m/s) precisely at the southern open ocean boundary nodes (e.g., nodes 12835, 8221, 12127, which all sit at exactly 12.000°N).
* **Root Cause**: This is a classic "cyclone crossing the open boundary" issue. As Amphan's eye passed over the 12°N boundary line from the south, SWAN calculated massive 150+ km/h hurricane wave stresses exactly on the boundary elements. Because the open boundary essentially has zero wave energy immediately outside the domain, this created a near-infinite gradient in wave radiation stress (`dSxx/dx`). The nonlinear momentum advection solver attempted to resolve this infinite gradient, leading to an infinite velocity loop and mathematical explosion.
* **Solution (3-Part Fix applied to `fort.15`)**:
  1. **Damped GWCE (`TAU0 = 1.0`)**: Set `TAU0` to `1.0` (highly diffusive). This allows the continuity equation to mathematically absorb extreme high-frequency shocks rather than blowing up instantly.
  2. **Disabled Momentum Advection (`NOLICA = 0`)**: Turned off nonlinear advection. Advection is notorious for blowing up at artificial open boundaries and contributes very little (<5%) to peak coastal storm surge, making this a highly recommended and safe engineering trade-off to ensure completion.
  3. **Reduced Time Step (`DTDP = 1`)**: Halved the time step from 2.0s to 1.0s to give the solver more time to resolve extreme velocity gradients without violating Courant stability rules.

