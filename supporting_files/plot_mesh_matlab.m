% plot_mesh_matlab.m
addpath('C:\Users\atikr\OceanMesh2D');
addpath('C:\Users\atikr\OceanMesh2D\utilities');

try
    disp('Loading mesh...');
    msh_obj = msh('C:\Users\atikr\OceanMesh2D\actual_amphan\fort.14');

    f = figure('Visible', 'off', 'Position', [100, 100, 1200, 1000]);
    
    disp('Plotting boundaries using OceanMesh2D native plot method...');
    plot(msh_obj, 'type', 'bd', 'proj', 'none');
    title('Amphan Storm Surge Mesh with Boundaries (OceanMesh2D Native)');

    disp('Saving image...');
    print(gcf, 'C:\Users\atikr\OceanMesh2D\actual_amphan\mesh_boundaries.png', '-dpng', '-r300');
    disp('Image saved successfully!');
catch ME
    disp('An error occurred:');
    disp(ME.message);
end
exit;
