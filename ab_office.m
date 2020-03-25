function [list_acc] = ab_office(options)
    % DEMO for testing MEDA on Office+Caltech10 datasets
    str_domains = {'Caltech10', 'amazon', 'webcam', 'dslr'};
    mode_new = 'closed';
    list_acc = []; 
    for i = 1 : 4
        for j = 1 : 4
            if i == j
                continue;
            end
            src = str_domains{i};
            tgt = str_domains{j};

            load(['data/' src '_SURF_L10.mat']);     % source domain
            fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); %每一维度做均�?
            Xs = zscore(fts,1);    clear fts   %标准化（归一化）
            Ys = labels;           clear labels

            load(['data/' tgt '_SURF_L10.mat']);     % target domain
            fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); %每一维度做均�?
            Xt = zscore(fts,1);     clear fts %标准化（归一化）
            Yt = labels;            clear labels
            % meda
            switch mode_new
                case 'closed'
                    [Acc,acc_iter_mmcd,~,~,MMD_distance_iter_mmcd] = MK_MMCD(Xs,Ys,Xt,Yt,options, mode_new, src, tgt);
                end
            fprintf('%s --> %s: %.2f accuracy \n\n', src, tgt, Acc * 100);
            list_acc = [list_acc; Acc];
        end
    end
end