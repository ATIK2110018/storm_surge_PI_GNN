addpath('C:\Users\atikr\OceanMesh2D');
addpath('C:\Users\atikr\OceanMesh2D\utilities');
m = msh('C:\Users\atikr\OceanMesh2D\actual_amphan\fort.14');
m = Make_f15(m, '16-May-2020 00:00', '24-May-2020 00:00', 2.0, 'NWS', 8, 'WTIMNC', 3, 'RSTIMNC', 3, 'const', {'M2', 'S2', 'N2', 'K1', 'O1'}, 'tidal_database', 'C:\TPXO_10_atlas_V2\TPXO10_atlas_v2_nc\h_**_tpxo10_atlas_30_v2.nc');
disp(['Size before write: ', num2str(size(m.f15.opealpha(1).val))]);
write(m, 'C:\Users\atikr\OceanMesh2D\actual_amphan\test_fort.15', 'f15');
disp('Done writing test_fort.15');
exit;
