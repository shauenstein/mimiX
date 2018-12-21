%% Read ncfile
data_folder = '\\ad.helsinki.fi\home\j\juzmakin\Documents\Freiburg\mimiX\MATLABcode\';

%% Plot covariates and response
covariates = {'CVid', 'Lat', 'Lon', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'};
env_cov = {'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'};

% matrix of covariates
x = [];
for i02 = 1:size(covariates,2)
    x(:,i02) = ncread('\\ad.helsinki.fi\home\j\juzmakin\Documents\Freiburg\mimiX\Rcode\dataset212.nc',covariates{i02});
end

% vector of y:s
y(:,:) = ncread('\\ad.helsinki.fi\home\j\juzmakin\Documents\Freiburg\mimiX\Rcode\dataset212.nc','y');

figure('pos', [100,100,500,300])

% create a matrix for display
ind_lon = linspace(0,1,50);
ind_lat = fliplr(linspace(0,1,50));

% loop through covariates
for i02 = 1:size(env_cov,2)
    sp(i02) = subplot(2,4,i02);
    ind_z = reshape(x(:,strcmp(covariates,env_cov{i02})),[50 50]);
    imagesc(ind_lon,ind_lat,ind_z'), shading flat
    title(env_cov{i02})
    axis equal
    set(gca, 'Clim', [-1 1], 'YLim', [0 1])
    if i02 == 5
        cb = colorbar('Location', 'westoutside');
        set(cb, 'pos', [.05, .12, .02, .3], 'Limits', [-1,1])
    end
end

% plot also response variable
sp(8) = subplot(2,4,8);
ind_z = reshape(y,[50 50]);
imagesc(ind_lon, ind_lat, ind_z'), shading flat
axis equal
set(gca, 'Clim', [-1 1], 'YLim', [0 1])


%% Build models with no prior information
% likelihood
lik = lik_gaussian;

% standardize covariates
standind = find(strcmp(covariates,env_cov{1})):size(x,2);
% mx = mean(x(:,standind)); stdx = std(x(:,standind));
% x_sub = [x(:,1:min(standind)-1) (x(:,standind)-mx)./stdx];

% intercept
cfc = gpcf_constant('constSigma2', 10, 'constSigma2_prior', prior_fixed);

% linear effect
cf_linear = gpcf_linear('selectedVariables', standind(4), ...
        'coeffSigma2', 10, 'coeffSigma2_prior', prior_fixed);

% squared effect
cf_squared = gpcf_squared('selectedVariables', standind(4), ...
        'coeffSigma2', 10, 'coeffSigma2_prior', prior_fixed);

cf_interaction = gpcf_prod('cf', {gpcf_linear('selectedVariables', ...
    standind(4), 'coeffSigma2', 10, 'coeffSigma2_prior', prior_fixed), ...
    gpcf_linear('selectedVariables', standind(3), 'coeffSigma2', 10, ...
    'coeffSigma2_prior', prior_fixed)});

cfs = gpcf_matern32('magnSigma2', 1, 'lengthScale', 2, 'selectedVariables', ...
    [2 3], 'lengthScale_prior', prior_t('s2',.1), 'magnSigma2_prior', prior_t('s2',1));

gp = gp_set('lik',lik,'cf',{cfc cf_linear cf_squared cf_interaction cfs});

opt=optimset('TolFun',1e-3,'TolX',1e-3,'Display','iter');
gp=gp_optim(gp,x,y,'opt',opt);

[Ef, Varf] = gp_pred(gp,x,y,x,'predcf',5);
[Ef_j, Covf_j] = gp_jpred(gp,x,y,x,'predcf',5);
LCovf_j = chol(Covf_j,'lower');
samp_gp = repmat(Ef_j,1,2000) + LCovf_j*randn(size(Ef_j,1),2000);

[Ef1,Varf1] = gp_pred(gp,x,y,[zeros(1,6) 1 zeros(1,3)],'predcf',1);
[Ef2,Varf2] = gp_pred(gp,x,y,[zeros(1,6) 1 zeros(1,3)],'predcf',2);
[Ef3,Varf3] = gp_pred(gp,x,y,[zeros(1,6) 1 zeros(1,3)],'predcf',3);
[Ef4,Varf4] = gp_pred(gp,x,y,[zeros(1,5) 1 1 zeros(1,3)],'predcf',4);
Beta_estimate = [Ef1, Varf1, Ef2, Varf2, Ef3, Varf3, Ef4, Varf4];

dlmwrite(sprintf('%sOmitted_pred.txt',data_folder),samp_gp);
dlmwrite(sprintf('%sOmitted_pred_mean.txt',data_folder),Ef);
dlmwrite(sprintf('%sBeta_estimate.txt',data_folder),Beta_estimate);
save([data_folder 'Model_gp'], 'gp', 'Ef', 'Ef1', 'Ef2', 'Ef3','Ef4')
