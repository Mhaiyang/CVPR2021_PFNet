clc;clear;

algorithms = {
%      '2017-CVPR-FPN';
%      '2017-CVPR-PSPNet';
%      '2017-CVPR-MaskRCNN';
%      '2018-DLMIA-UNet++';
%      '2018-CVPR-DSC';
%      '2018-CVPR-PiCANet';
%      '2018-ECCV-BDRAR';
%      '2019-CVPR-HTC';
%      '2019-CVPR-MSRCNN';
%      '2019-CVPR-BASNet';
%      '2019-CVPR-CPD_ResNet';
%      '2019-CVPR-PFANet';
%      '2019-ICCV-EGNet';
%      '2020-AAAI-F3Net';
%      '2020-AAAI-GCPANet';
%      '2020-MICCAI-PraNet';
%      '2020-CVPR-MINet-R';
%      '2020-CVPR-SINet';

     'PFNet';
    };

datasets = {
              'CHAMELEON';
              'CAMO';
              'COD10K';
              'NC4K';
    };

tic
for i = 1:numel(algorithms)
    alg = algorithms{i};
    fprintf('%s\n', alg);
    txt_path = ['./mat/' alg '/'];
    if ~exist(txt_path, 'dir'), mkdir(txt_path); end
    fileID = fopen([txt_path 'results.txt'],'w');
    
    for j = 1:numel(datasets)
        dataset      = datasets{j};
        predpath     = ['../results/' alg '/' dataset '/'];
        maskpath     = ['../../data/NEW/test/' dataset '/mask/'];
        if ~exist(predpath, 'dir'), continue; end

        names = dir(['~/data/NEW/test/' dataset '/mask/*.png']);
        names = {names.name}';
        wfm          = 0; mae    = 0; sm     = 0; fm     = 0; prec   = 0; rec    = 0; em     = 0;
        score1       = 0; score2 = 0; score3 = 0; score4 = 0; score5 = 0; score6 = 0; score7 = 0;

        results      = cell(numel(names), 6);
        ALLPRECISION = zeros(numel(names), 256);
        ALLRECALL    = zeros(numel(names), 256);
        file_num     = false(numel(names), 1);
        
        for k = 1:numel(names)
            name          = names{k,1};
            results{k, 1} = name;
            file_num(k)   = true;
            fgpath        = [predpath name];
            fg            = imread(fgpath);

            gtpath = [maskpath name];
            gt = imread(gtpath);

            if length(size(fg)) == 3, fg = fg(:,:,1); end
            if length(size(gt)) == 3, gt = gt(:,:,1); end
            fg = imresize(fg, size(gt)); 
            fg = mat2gray(fg); 
            gt = mat2gray(gt);
            
            gt(gt>=0.5) = 1; gt(gt<0.5) = 0; gt = logical(gt);
            score1                   = MAE(fg, gt);
            score5                   = wFmeasure(fg, gt); 
            score6                   = Smeasure(fg, gt);
            score7                   = Emeasure(fg, gt);
            mae                      = mae  + score1;
            wfm                      = wfm  + score5;
            sm                       = sm   + score6;
            em                       = em   + score7;
            results{k, 2}            = score6; 
            results{k, 3}            = score7; 
            results{k, 4}            = score5; 
            results{k, 5}            = score1;

        end

        file_num = double(file_num);
        fm       = fm  / sum(file_num);
        mae      = mae / sum(file_num); 
        wfm      = wfm / sum(file_num); 
        sm       = sm  / sum(file_num); 
        em       = em  / sum(file_num);
        fprintf(fileID, '%10s (%4d images): S:%6.3f, E:%6.3f, F:%6.3f, M:%6.3f\n', dataset, sum(file_num), sm, em, wfm, mae);
        fprintf('%10s (%4d images): S:%6.3f, E:%6.3f, F:%6.3f, M:%6.3f\n', dataset, sum(file_num), sm, em, wfm, mae);
        save_path = ['./mat' filesep alg filesep dataset filesep];
        if ~exist(save_path, 'dir'), mkdir(save_path); end
        save([save_path 'results.mat'], 'results');

    end
end
toc
