% DEMO for testing MEDA on Office+Caltech10 datasets
str_domains = {'Caltech10', 'amazon', 'webcam', 'dslr'};
list_acc = [];
mode_new = 'jda';
for i = 1 : 4
    for j = 1 : 4
        if i == j
            continue;
        end
        if i~= 4
            continue;
        end
        if j ~= 3
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
            case 'jda'
                options.dim = 100;
                options.rho = 1.0;
                options.p = 10;
                options.lambda = 1.0;
                options.eta = 0.1;
                options.T = 10;
                options.gamma = 1.0;
                options.mu = 1;
                options.kernel_type = 'linear';
                [Acc,~,~] = JDA(Xs,Ys,Xt,Yt,options);
            case 'closed'
                options.d = 20;
                options.rho = 1.0;
                options.p = 10;
                options.lambda = 10.0;
                options.eta = 0.1;
                options.T = 10;
                options.gamma = 0.1;
                options.mu = 1;
        %         [Acc,~,~,~] = MEDA(Xs,Ys,Xt,Yt,options);
                [Acc,~,~,~] = MSDI(Xs,Ys,Xt,Yt,options, mode_new, src, tgt);
            case 'adam'
                options.d = 20;
                options.rho = 1.0;
                options.p = 10;
                options.lambda = 10.0;
                options.eta = 0.1;
                options.T = 100;
                options.gamma = 0.1;
                options.mu = 1;
                options.add = 0.1;  % 0.01
        %         [Acc,~,~,~] = MEDA(Xs,Ys,Xt,Yt,options);
                [Acc,~,~,~] = MSDI_sgd(Xs,Ys,Xt,Yt,options, mode_new, src, tgt);
            case 'init'
                options.d = 20;
                options.rho = 1.0;
                options.p = 10;
                options.lambda = 10.0;
                options.eta = 0.1;
                options.T = 10;
                options.gamma = 0.1;
                options.mu = 1;
        %         [Acc,~,~,~] = MEDA(Xs,Ys,Xt,Yt,options);
                [Acc,~,~,~] = MSDI(Xs,Ys,Xt,Yt,options, mode_new, src, tgt);
            case 'plot'
                    options.d = 20;
                    options.rho = 1.0;
                    options.p = 10;
                    options.lambda = 10.0;
                    options.eta = 0.1;
                    options.T = 100;
                    options.gamma = 0.1;
                    options.mu = 1;
                    options.add = 0.01;  % 0.01
                    [Acc,~,~,~,cc_meda_t] = MEDA_plot(Xs,Ys,Xt,Yt,options);
                    [Acc,~,~,~,cc_source_only_s, cc_source_only_t,cc_sgd_t] = MSDI_sgd_plot(Xs,Ys,Xt,Yt,options, mode_new, src, tgt);
                    cc_meda_t = roundn(cc_meda_t,-2);
                    cc_source_only_s =  roundn(cc_source_only_s,-2);
                    cc_source_only_t =  roundn(cc_source_only_t,-2);
                    cc_sgd_t =  roundn(cc_sgd_t,-2);
                    xray = {'backpack', 'touring-bike','calculator','head-phones','keyboard','laptop','monitor','mouse','coffeemug','projector'};
                    yray = {'backpack', 'touring-bike','calculator','head-phones','keyboard','laptop','monitor','mouse','coffeemug','projector'};
%                     xray = {'1','2','3','4','5','6','7','8','9','10'};
%                     yray = {'1','2','3','4','5','6','7','8','9','10'};
                    figure;
                    heatmap(xray,yray,cc_source_only_s);
                    figure;
                    heatmap(xray,yray,cc_source_only_t);
                    figure;
                    heatmap(xray,yray,cc_sgd_t);
                    figure;
                    heatmap(xray,yray,cc_meda_t);

                case 'init_plot'
                    options.d = 20;
                    options.rho = 0.0;
                    options.p = 10;
                    options.lambda = 0.0;
                    options.eta = 1;
                    options.T = 10;
                    options.gamma = 0.1;
                    options.mu = 1;
                    options.delta = 0.0;
            %         [Acc,~,~,~] = MEDA(Xs,Ys,Xt,Yt,options);
                    [Acc,~,~,~] = MSDI(Xs,Ys,Xt,Yt,options, mode_new, src, tgt);
            end
        fprintf('%s --> %s: %.2f accuracy \n\n', src, tgt, Acc * 100);
        list_acc = [list_acc; Acc];
    end
end
