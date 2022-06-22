data_dir = "/home/gatavins/brain_age/data";
bct_dir = "/home/gatavins/BCT/2019_03_03_BCT";
addpath(data_dir,bct_dir);
subjects = readtable(strcat(data_dir,"/func_fcon-parcWITHAGE-320x55279_20220615.csv"));
atlas_parcID = fopen(strcat(data_dir,"/gordon333CommunityAffiliation.1D"));
atlas_parc = textscan(atlas_parcID,"%u");
atlas_communities = double(atlas_parc{1})';

% preparing table for results
cols = ["sub","age",...
    genvarname(repelem("strength",333),"strength"),...
    genvarname(repelem("clustering",333),"clustering"),...
    genvarname(repelem("part_coef_avg",333),"part_coef_avg"),...
    "avgweight","glob_clustering","glob_participation",...
    "modularity","optim_numcom"];
colcount = width(cols);
rowcount = height(subjects);
vartypes = ["string",string(repelem({'double'},1005))];
size = [rowcount colcount];
results = table('Size',size,'VariableTypes',vartypes,...
        'VariableNames',cols);

for n=1:height(subjects)
% storing id, age, and FC vector
results(n,1) = subjects(n,1);
results(n,2) = subjects(n,end);
vect = table2array(subjects(n,2:end-1));

% reshaping vectors into 333 x 333 symmetric matrices
fcmat = zeros(333);
fcmat(triu(ones(333),1)==1) = vect;
fcmat(tril(ones(333),-1)==1) = fcmat(triu(ones(333),1)==1)';
zfcmat = atanh(fcmat); % Fisher Z-transform

% node-level measures
    % strength - positive & negative average per node
    [pos, neg, ~, ~] = strengths_und_sign(zfcmat);
    node_strength_mean = mean([pos',neg'],2);
    results{n,3:335} = node_strength_mean';
    
    % clustering coefficient - Constantini & Perugini (one value per node) 
    results{n,336:668} = clustering_coef_wu_sign(zfcmat,3)';

    % participation coefficient with Gordon atlas, averaging pos & neg
    [pcoef_pos, pcoef_neg] = participation_coef_sign(zfcmat,atlas_communities);
    pcoef_avg = mean([pcoef_pos, pcoef_neg],2);
    results{n,669:1001} = pcoef_avg';

% global measures
    results{n,1002} = mean(vect); % non-zero weights (all weights ~=0)
    [~, ~, glob_pos, glob_neg] = clustering_coef_wu_sign(zfcmat,2);
    results{n,1003} = ((glob_pos + glob_neg) / 2); % avg clustering coef
    results{n,1004} = ((mean(pcoef_pos) + mean(pcoef_neg)) / 2); % avg PC

    % modularity measures (Louvain with negative asym & quality index)
    % using 'negative_asym' as per Rubinov & Sporns (2011) recommendation
    modvector = zeros(2,100);
    for k=1:100
         [C, Q] = community_louvain(zfcmat,1,[],'negative_asym');
         modvector(1,k) = max(C);
         modvector(2,k) = Q;
    end
    mod_mean = mean(modvector,2);
    results{n,1005} = mod_mean(2,1);
    results{n,1006} = mod_mean(1,1);
end

writetable(results,strcat(data_dir,"/graph_measures_signed.csv"));
