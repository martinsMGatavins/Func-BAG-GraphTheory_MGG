% Graph construction & measure extraction
% author: Martins M Gatavins
% Dependencies: Brain Connectivity Toolbox (BCT)

% Virtual
data_dir = "/home/gatavins/brain_age/data";
bct_dir = "/home/gatavins/BCT/2019_03_03_BCT";

% Local
% data_dir = "~/Desktop/BIDS_WUSTL_2022/second_set/data";
% bct_dir = "~/Desktop/BIDS_WUSTL_2022/BCT/2019_03_03_BCT";

% Adding paths and reading in files
addpath(data_dir,bct_dir);
subjects = readtable(strcat(data_dir,"/func_fcon-parc-888x55278_filtered.csv"));
atlas_parcID = fopen(strcat(data_dir,"/gordon333CommunityAffiliation.1D"));
atlas_parc = textscan(atlas_parcID,"%u");
atlas_communities = double(atlas_parc{1})';

% Initializing table for outputs
cols = ["sub_method",genvarname(repelem("strength",333),"strength"),...
    genvarname(repelem("clustering",333),"clustering"),...
    genvarname(repelem("part_coef",333),"part_coef"),...
    genvarname(repelem("nodal_eff",333),"nodal_eff"),...
    genvarname(repelem("bcentr",333),"bcentr"),...
    genvarname(repelem("intrafc",333),"intrafc"),...
    genvarname(repelem("interfc",333),"interfc"),...
    "avg_edgeweight","modularity_louvain"];
colcount = width(cols);
rowcount = height(subjects);
vartypes = ["string",string(repelem({'double'},333*7+2))];
size = [rowcount colcount];
results = table('Size',size,'VariableTypes',vartypes,...
        'VariableNames',cols);
    
for i=1:height(results) % or testing value (like 2)
results(i,1) = subjects(i,2);
    
vect = table2array(subjects(i,3:end));

% reshaping vectors into 333 x 333 symmetric matrices
fcmat = zeros(333);
fcmat(triu(ones(333),1)==1) = vect;
fcmat(tril(ones(333),-1)==1) = fcmat(triu(ones(333),1)==1)';

% Fisher Z-transform (keeping raw for averaging functional connectivities)
zfcmat_raw = atanh(fcmat);

% thresholding top 20% strongest edges
zfcmat = threshold_proportional(zfcmat_raw,0.25);

if sum(degrees_und(zfcmat)==0)>0
    disp("Disconnected graph")
    continue
end
disp("Calculating nodal measures")
% node strength calculation
results{i,2:334} = strengths_und(zfcmat);

% clustering coefficient
results{i,335:667} = clustering_coef_wu(zfcmat)';

% participation coefficient
results{i,668:1000} = participation_coef(zfcmat,atlas_communities)';

% nodal efficiency
[geff, neff, leff] = rout_efficiency(zfcmat,'inv');
leff = 1./leff;
results{i,1001:1333} = leff';

% betweenness centrality
results{i,1334:1666} = (betweenness_wei(zfcmat)/((332*331)/2))';

disp("Calculating network functional connectivities")
% intra & inter-network connectivity
% code adapted from Ursula Tooley 2022
intra = zeros(1,333);
inter = zeros(1,333);
for c=1:length(unique(atlas_communities))
    Cnodes = find(atlas_communities == c);
    for d=1:sum(atlas_communities==c)
        Wi = atlas_communities(:) == c;
        Bi = atlas_communities(:) ~= c;
        curr = Cnodes(d);
        Wconn = zfcmat_raw(curr,Wi);
        Bconn = zfcmat_raw(curr,Bi);
        intra(curr) = mean(Wconn(Wconn~=0));
        inter(curr) = mean(Bconn);
    end
end

results{i,1667:1999} = intra;
results{i,2000:2332} = inter;

disp("Calculating the last few non-nodal measures")
% Average edge weight
results{i,2333} = mean(zfcmat(zfcmat~=0));

% Modularity of Louvain-generated partition
modvector = zeros(1,100);
    for k=1:100
         [C, Q] = community_louvain(zfcmat,1,[]);
         modvector(1,k) = Q;
    end
results{i,2334} = mean(modvector,2);
end

writetable(results,"/home/gatavins/brain_age/graph_data/graph25perc-fcon_888x2334_20220721.csv");
