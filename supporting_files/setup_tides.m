addpath('C:\Users\atikr\OceanMesh2D');
addpath('C:\Users\atikr\OceanMesh2D\utilities');
addpath('C:\Users\atikr\OceanMesh2D\m_map');

try
    disp('Loading mesh...');
    msh = msh('C:\Users\atikr\OceanMesh2D\actual_amphan\fort.14');

    % Setup the requested tidal constituents
    consts = {'m2', 's2', 'n2', 'k1', 'o1'};
    msh.f15.nbfr = length(consts);
    for i=1:length(consts)
        msh.f15.opealpha(i).name = consts{i};
        msh.f15.bountag(i) = 1;
    end

    disp('Extracting TPXO data...');
    tpxo_pattern = 'C:\TPXO_10_atlas_V2\TPXO10_atlas_v2_nc\h_**_tpxo10_atlas_30_v2.nc';
    msh = tidal_data_to_ob(msh, tpxo_pattern, consts);

    disp('Harmonic constituents extracted successfully.');
    
    % Save the msh object
    save('C:\Users\atikr\OceanMesh2D\actual_amphan\msh_with_tides.mat', 'msh', '-v7.3');
    disp('Saved to msh_with_tides.mat');

catch ME
    disp('Error occurred:');
    disp(ME.message);
end

exit;
