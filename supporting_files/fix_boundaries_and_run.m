addpath('C:\Users\atikr\OceanMesh2D');
addpath('C:\Users\atikr\OceanMesh2D\utilities');

disp('Loading mesh...');
m = msh('C:\Users\atikr\OceanMesh2D\actual_amphan\fort.14');

% Temporarily assign TPXO tides to see which nodes are NaN
ts = '16-May-2020 00:00';
te = '24-May-2020 00:00';
dt = 2.0; 
tpxo_pattern = 'C:\TPXO_10_atlas_V2\TPXO10_atlas_v2_nc\h_**_tpxo10_atlas_30_v2.nc';
consts = {'M2', 'S2', 'N2', 'K1', 'O1'};

disp('Evaluating tides to find invalid nodes...');
m_test = Make_f15(m, ts, te, dt, 'NWS', 8, 'WTIMNC', 3, 'RSTIMNC', 3, 'const', consts, 'tidal_database', tpxo_pattern);

% Find which boundary nodes are NaN in O1
nan_idx = isnan(m_test.f15.opealpha(5).val(:,1));
disp(['Found ', num2str(sum(nan_idx)), ' NaN nodes on open boundaries.']);

if sum(nan_idx) > 0
    % Find the original node IDs of the NaN nodes
    % m_test.op.nbdv contains the node IDs for each boundary
    nan_nodes = [];
    count = 1;
    for n = 1:m.op.nope
        for i = 1:m.op.nvdll(n)
            if nan_idx(count)
                nan_nodes(end+1) = m.op.nbdv(i,n);
            end
            count = count + 1;
        end
    end
    
    disp('NaN nodes are:');
    disp(nan_nodes);
    
    % Now, instead of manually rewriting boundaries, let's just 
    % fill the NaN values with nearest neighbor extrapolation!
    % Wait! User said "dont use the nodes as boundary where you didnot found data."
    % We MUST change them to land nodes.
    % To do this safely without breaking the msh object, we can re-run make_bc 
    % but supply a custom depth or mask, OR just manually edit the boundary cell array.
    % Let's manually edit m.bd.cell!
    
    for i = 1:length(nan_nodes)
        node = nan_nodes(i);
        % Find which open boundary has this node and remove it
        for j = 1:length(m.bd.cell)
            if m.bd.ibtype(j) == 0 % Open boundary
                idx = find(m.bd.cell{j} == node);
                if ~isempty(idx)
                    m.bd.cell{j}(idx) = [];
                    % Add it to a land boundary? 
                    % If it's at the end, the adjacent land boundary will pick it up?
                    % Actually, we can just create a new land boundary segment for these nodes!
                    m.bd.cell{end+1} = [node];
                    m.bd.ibtype(end+1) = 21;
                    m.bd.bountag(end+1).name = 'Land';
                    m.bd.bountag(end+1).val = 0;
                end
            end
        end
    end
    
    % Rebuild m.op and m.ld
    m = setup_bcs(m);
    
    disp('Writing new fort.14...');
    write(m, 'C:\Users\atikr\OceanMesh2D\actual_amphan\fort.14', 'f14');
    
    disp('Regenerating fort.15...');
    m = Make_f15(m, ts, te, dt, 'NWS', 8, 'WTIMNC', 3, 'RSTIMNC', 3, 'const', consts, 'tidal_database', tpxo_pattern, 'namelist', {'swanoutput', 'nws8'});
    write(m, 'C:\Users\atikr\OceanMesh2D\actual_amphan\fort.15', 'f15');
    disp('Fixed mesh and control files generated successfully!');
else
    disp('No NaN nodes found!');
end
exit;
