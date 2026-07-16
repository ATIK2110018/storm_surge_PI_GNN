% generate_fort15.m

addpath('C:\Users\atikr\OceanMesh2D');
addpath('C:\Users\atikr\OceanMesh2D\utilities');

try
    disp('Loading fresh mesh...');
    msh_obj = msh('C:\Users\atikr\OceanMesh2D\actual_amphan\fort.14');

    disp('Configuring ADCIRC parameters and extracting tides natively...');
    ts = '16-May-2020 00:00';
    te = '24-May-2020 00:00';
    dt = 2.0; % Global time step in seconds
    
    tpxo_pattern = 'C:\TPXO_10_atlas_V2\TPXO10_atlas_v2_nc\h_**_tpxo10_atlas_30_v2.nc';
    consts = {'M2', 'S2', 'N2', 'K1', 'O1'};

    % Make_f15 will extract the tides and build the full control file
    msh_obj = Make_f15(msh_obj, ts, te, dt, 'NWS', 8, 'WTIMNC', 3, 'RSTIMNC', 3, 'const', consts, 'tidal_database', tpxo_pattern, 'namelist', {'swanoutput', 'nws8'});

    disp('Writing fort.15 file...');
    write(msh_obj, 'C:\Users\atikr\OceanMesh2D\actual_amphan\fort.15', 'f15');
    
    disp('fort.15 generated successfully!');
catch ME
    disp('An error occurred during generation:');
    disp(ME.message);
end
exit;
