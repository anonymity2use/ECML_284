[y,xt] = libsvmread('heart_scale');
model=train(y, xt)
[l,a]=predict(y, xt, model);



str_domains = {'1', '2'};
list_acc = [];
mode_new = 'closed';
for i = 1 : 2
    src = str_domains{i};
    tar = num2str(3 - i);

    load(['data/COL20/COIL_' src '.mat']);     

    % source domain
    X_src = X_src ./ repmat(sum(X_src, 1), size(X_src,1),1);
    Xs = zscore(X_src, 1); clear X_src
    Ys = Y_src;            clear Y_src

    % target domain
    X_tar = X_tar ./ repmat(sum(X_tar, 1), size(X_tar,1),1);
    Xt = zscore(X_tar, 1); clear X_tar
    Yt = Y_tar;            clear Y_tar

    % meda
    switch mode_new
        case 'closed'
            options.d = 20;
            options.rho = 0.1;
            options.p = 10;
            options.lambda = 10.0;
            options.eta = 0.1;
            options.T = 15;
            options.gamma = 0.1;
            options.mu = 0.6;
            options.delta = 0.01;
            options.dim = 100;    
            options.kernel_type = 'linear';    
            model=train(Ys, Xs')
            [l,a]=predict(Yt, Xt', model);
            %[Acc,~,~,~] = MEDA(Xs',Ys,Xt',Yt,options);
            %[Acc,~,~,~] = MK_MMCD(Xs',Ys,Xt',Yt,options,mode_new,src,tar);
        case 'adam'
            options.d = 20;
            options.rho = 0.1;
            options.p = 10;
            options.lambda = 90.0;
            options.eta = 0.05;
            options.T = 1000;
            options.gamma = 0.1;
            options.mu = 0.6;
            options.delta = 0.01;
            options.add = 0.01;
        %     [Acc,~,~,~] = MEDA(Xs',Ys,Xt',Yt,options);
            [Acc,~,~,~] = MK_MMCD_sgd(Xs',Ys,Xt',Yt,options,mode_new,src,tar);
        case 'init'
            options.d = 20;
            options.rho = 0.1;
            options.p = 10;
            options.lambda = 10.0;
            options.eta = 0.1;
            options.T = 15;
            options.gamma = 0.1;
            options.mu = 0.6;
            options.delta = 0.01;
            options.add = 0.01;
        %     [Acc,~,~,~] = MEDA(Xs',Ys,Xt',Yt,options);
            [Acc,~,~,~] = MK_MMCD(Xs',Ys,Xt',Yt,options,mode_new,src,tar);
        case 'init_zero'
            options.d = 20;
            options.rho = 0.0;
            options.p = 10;
            options.lambda = 0.0;
            options.eta = 0.1;
            options.T = 5;
            options.delta = 0.0;
            options.add = 0.0;
        %     [Acc,~,~,~] = MEDA(Xs',Ys,Xt',Yt,options);
            [Acc,~,~,~] = MK_MMCD(Xs',Ys,Xt',Yt,options,mode_new,src,tar);
        case 'plot'
            options.d = 20;
            options.rho = 0.1;
            options.p = 10;
            options.lambda = 10.0;
            options.eta = 0.1;
            options.T = 15;
            options.gamma = 0.1;
            options.mu = 0.6;
            options.delta = 0.01;
            options.add = 0.01;
            [Acc,~,~,~,cc_meda_t] = MEDA_plot(Xs',Ys,Xt',Yt,options);
%             [Acc,~,~,~,cc_source_only_s, cc_source_only_t,cc_sgd_t] = MK_MMCD_sgd_plot(Xs',Ys,Xt',Yt,options,mode_new,src,tar);
            xray = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'};
            yray = {'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'};
        case 'init_plot'
            options.d = 20;
            options.rho = 0.0;
            options.p = 10;
            options.lambda = 0.0;
            options.eta = 0.1;
            options.T = 3;
            options.gamma = 0.1;
            options.mu = 0.6;
            options.delta = 0.0;
            options.add = 0.0;
        %     [Acc,~,~,~] = MEDA(Xs',Ys,Xt',Yt,options);
            [Acc,~,~,~] = MK_MMCD(Xs',Ys,Xt',Yt,options,mode_new,src,tar);
           
    end
    fprintf('COIL_%s -> %s :%.2f accuracy \n\n', src, tar,Acc * 100);
    list_acc = [list_acc; Acc];
end

