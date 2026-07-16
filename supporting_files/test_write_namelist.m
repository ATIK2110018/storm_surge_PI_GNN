addpath('C:\Users\atikr\OceanMesh2D');
addpath('C:\Users\atikr\OceanMesh2D\utilities');
m = msh('C:\Users\atikr\OceanMesh2D\actual_amphan\fort.14');
m = Make_f15(m, '16-May-2020 00:00', '19-May-2020 09:00', 2.0, 'NWS', 8, 'WTIMNC', [2020 5 16 0 1 1.0], 'RSTIMNC', 3, 'const', {'M2', 'S2', 'N2', 'K1', 'O1'}, 'tidal_database', 'C:\TPXO_10_atlas_V2\TPXO10_atlas_v2_nc\h_**_tpxo10_atlas_30_v2.nc', 'namelist', {'swanoutput', 'nws8'});
disp(['Size before write: ', num2str(size(m.f15.opealpha(1).val))]);
write(m, 'C:\Users\atikr\OceanMesh2D\actual_amphan\fort.15', 'f15');
disp('Done writing fort.15');
exit;
