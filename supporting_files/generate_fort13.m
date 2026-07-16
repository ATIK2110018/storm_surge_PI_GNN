% generate_fort13.m
% This script creates a depth-dependent Manning's n roughness file (fort.13)

addpath('C:\Users\atikr\OceanMesh2D');
addpath('C:\Users\atikr\OceanMesh2D\utilities');

try
    disp('Loading mesh...');
    msh_obj = msh('C:\Users\atikr\OceanMesh2D\actual_amphan\fort.14');
    
    disp('Calculating depth-dependent Manning''s n...');
    % In ADCIRC fort.14, depth is usually positive downwards.
    % msh_obj.b gives the depth array.
    depth = msh_obj.b;
    
    % Initialize Manning's n array with a default open-ocean value
    n_val = 0.020 * ones(size(depth));
    
    % Shallow water shelf (e.g., depth between 0 and 10 m)
    n_val(depth > 0 & depth <= 10) = 0.025;
    
    % Coastal and Overland areas (elevation above sea level, depth <= 0)
    % Assign a much higher roughness for land/mangroves
    n_val(depth <= 0) = 0.050; 
    
    disp('Applying ''Mn'' nodal attribute...');
    % Use Calc_f13 to assign the Manning's n ('Mn') array
    msh_obj = Calc_f13(msh_obj, 'Mn', 'assign', n_val);
    
    disp('Writing fort.13 file...');
    write(msh_obj, 'C:\Users\atikr\OceanMesh2D\actual_amphan\fort.13', 'f13');
    
    disp('fort.13 generated successfully!');
catch ME
    disp('Error occurred:');
    disp(ME.message);
end
exit;
