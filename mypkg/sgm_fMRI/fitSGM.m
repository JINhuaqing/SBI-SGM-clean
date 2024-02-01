function [theta, FC_pred, FX_pred, FC_emp, FX_emp] = fitSGM(SC, TS, params)
%fitSGM models fMRI timeseries data using the Spectral Graph Model
%   
%   Inputs:
%       SC = A Structural Connectivity Matrix
%       TS = A time series (Must be Time X Region)
%       params = may be one of two things:
%           1. A structure with some or all of following hyperparameters:
%               params.TR = The signals TR (the sampling interval in seconds)
%               params.fband = A two element vector where the first element
%                   is the low frequency and the second is the high frequency
%                   for the model to fit to and filter the data before fitting.
%               params.pwelch_windows = the number of windows to use for FX
%                   estimation in pwelch
%               params.costtype = options: 'corr' or 'mse'
%               params.perc_thresh = (true or false) whether to use a
%                   percolation threshold.
%               params.eig_weights = (true or false) whether to use eig
%                   weightings from the empirical function
%               params.deconvHRF = (true or false) whether to deconvolve
%                   the Hemodynamic Response Function from the TS
%               params.optimization = options: 'fmincon' or 'annealing'
%               params.model_focus = options: 'both' or 'FC' or 'FX'
%               params.fitmean = (true or false) whether the model should
%                   fit to mean spectra across regions opposed to all regions.
%               params.theta = a two element vector [alpha, tau] that
%                   specifies specific SGM parameters to evaluate. If this
%                   field is provided, then no fitting will occur!
%           2. A scalar representing the TR (the sampling interval in 
%               seconds) of the time series. In this case, default settings
%               will be used.
% 
%   Ouputs:
%       theta = a two element vector of output parameters [alpha, tau]
%       FC_pred = The SGM predicted FC matrix.
%       FX_pred = The SGM predicted FX (power spectral density).
%       FC_emp = The empirical FC used for fitting.
%       FX_emp = The empirical FX used for fitting.
% 
%   Usage: [theta, FC_pred, FX_pred, FC_emp, FX_emp] = fitSGM(SC, TS, params)
% 
%   Written by Ben Sipes (June 2023), methodology developed by Ashish Raj
%   circa 2021.

% % % % Function start:


% % % % Compute Laplacian and find its eigenmodes:
    SC = SC./sum(SC(:));
    rd = (sum(SC, 2)).';
    cd = sum(SC, 1);
    nroi = length(rd);
    L = eye(nroi) - diag(1./(sqrt(rd)+eps)) * SC* diag(1./(sqrt(cd)+eps));

    [U, ev] = eig(L);
    ev = diag(ev);
    [~, ii] = sort(ev, 'ascend');
    ev = ev(ii);
    U = U(:,ii);

% % % % Verify that the number of regions in TS and SC are the same:
    TS(isnan(TS)) = 0;
    sz_TS = size(TS);

    if ~isequal(sz_TS(2), nroi) && ~isequal(sz_TS(1), nroi)
        error('The number of regions in the timeseries does not equal the number of regions in the structure!');
    elseif isequal(sz_TS(2), nroi)
        nt = sz_TS(1);
    else
        TS = TS';
        nt = sz_TS(1);
    end

% % % % Check the params variable:
    if isstruct(params)
        
        if ~isfield(params,'TR')
            error("This function REQUIRES you add your signal's sampling rate (TR)!!!");
        end
        if ~isfield(params,'fband')
            params.fband = [0.01, 0.1];
        end
        if ~isfield(params,'pwelch_windows')
            params.pwelch_windows = [];
        else
            params.pwelch_windows = round(nt/(params.pwelch_windows));
        end
        if ~isfield(params,'costtype')
            params.costtype = 'corr';
        end
        if ~isfield(params,'perc_thresh')
            params.perc_thresh = false;
        end
        if ~isfield(params,'eig_weights')
            params.eig_weights = true;
        end
        if ~isfield(params,'deconvHRF')
            params.deconvHRF = false;
        end
        if ~isfield(params,'optimization')
            params.optimization = 'fmincon';
        end
        if ~isfield(params,'model_focus')
            params.model_focus = 'both';
        end
        if ~isfield(params,'fitmean')
            params.fitmean = false;
        end
        if isfield(params,'theta')
            theta = params.theta;
            forward_only = true;
        else
            forward_only = false;
        end
    elseif isscalar(params)
        TR = params; clear params;
        params.TR = TR;
        params.fband = [0.01, 0.1];
        params.pwelch_windows = [];
        params.costtype = 'corr';
        params.perc_thresh = false;
        params.eig_weights = true;
        params.deconvHRF = false;
        params.optimization = 'fmincon';
        params.model_focus = 'both';
        params.fitmean = false;
        forward_only = false;
    end


% % % % SGM Setup:

    if (nt < 64) && (strcmp(params.model_focus, 'both') || strcmp(params.model_focus, 'FX'))
        warning('Not enough timepoints for a good FFT; therefore SGM is only fitting to FC.')
        params.model_focus = 'FC';
    elseif nt < 128
        nfft = 64;
    else
        nfft = 128;
    end
    fvec = linspace(params.fband(1), params.fband(2), nfft);
    omegavec = 2*pi.*fvec(:);
    TR = params.TR;
    Fs = 1/TR;

% % % % % TS processing:

    % Demean
    TS = bsxfun(@minus, TS, mean(TS,1)); %demean
    % Detrend
    TS = detrend(TS);
    % Bandpass
    TS = lowpass(TS, params.fband(2), Fs);

    if params.deconvHRF
        TS = deconv_HRF(TS, TR);
    end

    % Pearson's correlation matrix
    FC_emp = corr(TS).*~eye(nroi); 
    FC_emp(isnan(FC_emp)) = 0;

    % Apply Percolation Threshold
    if params.perc_thresh
        FC_emp = perc_thresh(FC_emp);
    end
    
    % Ensure Symmetry
    FC_emp = triu(FC_emp,1) + triu(FC_emp)';

    % get a list of non-zero edges in FC
    nzinds = find(triu(FC_emp,1));

    % Compute Frequency
    FX_emp = sqrt( pwelch(TS, params.pwelch_windows, [], fvec, Fs) ); 
    FX_emp(isnan(FX_emp)) = 0;

    % Find the mean maximum amplitude frequency in empirical spectra to use
    % for FC modeling:
    [~, max_idx] = max(FX_emp); 
    f_at_max = fvec(max_idx);
    omega = 2*pi*mean(f_at_max);

    if params.eig_weights
        ev_weight = abs(diag(U'*FC_emp*U));
    else
        ev_weight = ones(nroi,1);
    end
    ev_weight(1) = 0;

% % % % % Set Optimization parameters:
    maxiter = 1000;

    costtype = params.costtype;
    fitmean = params.fitmean;

    theta0 = [0.5, 1];
    ll = [0.1, 0.1];
    ul = [10, 5];

    if strcmp(params.model_focus, 'both')
        objective_fxn = @myfun_both;
    elseif strcmp(params.model_focus, 'FC')
        objective_fxn = @myfun_FC;
    elseif strcmp(params.model_focus, 'FX')
        objective_fxn = @myfun_FX;
    else
        error('Invalid model focus specified...')
    end

% % % % OPTIMIZATION

if ~forward_only
    if strcmp(params.optimization, 'fmincon')
        theta = fmincon(objective_fxn, theta0, [], [], [], [], ll, ul, [], optimoptions('fmincon','Display','none','MaxIter', maxiter));
    elseif strcmp(params.optimization, 'annealing')
        theta = simulannealbnd(objective_fxn, theta0, ll, ul, optimoptions('simulannealbnd', 'Display','off','MaxIter', maxiter));
    else
        error('Invalid optimization function was specified...');
    end
end

FC_pred = ForwardModel_FC(theta);
FX_pred = ForwardModel_FX(theta);

%% Internal Functions for optimization:

function [err_both, r_FC, r_FX] = myfun_both(theta_star)
    [err_FC, r_FC] = myfun_FC(theta_star);
    [errvec, rvec] = myfun_FX(theta_star);
    r_FX = nanmean(rvec);
    err_FX = nanmean(errvec);

    err_both = err_FC + err_FX;
end

% FUNCTIONAL CONNECTIVITY SGM PREDICTION:
function [err, r] = myfun_FC(theta_star)
    outFC = ForwardModel_FC(theta_star);

    r = corr(FC_emp(nzinds), outFC(nzinds));
    err = abs(1-r); 

end

function outFC = ForwardModel_FC(theta_star)
% This is the forward SCFC eigen model, that gives predicted FC from SC, and
% returns the best Pearson R between model and empirical FC
    alpha = tanh(theta_star(1)); 
    tau = theta_star(2);

    He = 1/tau^2./(1i*omega+1/tau).^2;
    newev = 1 ./ (1i*omega + 1/tau*He*(1- alpha*(1 - ev)));
    newev = (abs(newev)).^2 .*ev_weight;

    outFC = U * bsxfun(@times, newev(:), U');
    dg = 1./(1e-4+sqrt(diag(outFC)));
    outFC = bsxfun(@times, outFC, dg);
    outFC = bsxfun(@times, outFC, dg.');

end

% BOLD PSD SGM PREDICTION:
function [errvec, rvec] = myfun_FX(theta_star)
    outFx = ForwardModel_FX(theta_star);  % evaluate model at params
    rvec = nan(nroi,1);

    if fitmean
        qdata = abs(mean(FX_emp,2));
        qmodel = abs(mean(outFx,2));
        
        switch costtype
            case 'corr'
                rvec = corr(qdata, qmodel, 'type', 'Pearson', 'rows', 'complete');
                errvec = abs(1 - rvec);
            case 'mse'
                qdata(isnan(qmodel)) = [];
                qmodel(isnan(qmodel)) = [];
                qdata = rescale(qdata);
                qmodel = rescale(qmodel);
                rvec = corr(qdata, qmodel, 'type', 'Pearson', 'rows', 'complete');
                rvec(isnan(rvec)) = 0;
                errvec = immse(qdata, qmodel);
        end
    else
        switch costtype
            case 'corr'
                for n = 1:nroi
                    qdata = abs(FX_emp(:, n));
                    qmodel = abs(outFx(:,n));
                    rvec(n) = corr(qdata, qmodel, 'type', 'Pearson', 'rows', 'complete');
                end

                rvec(isnan(rvec)) = 0;
                errvec = abs(1 - rvec);

            case 'mse'
                errvec = nan(nroi,1);
                for n = 1:nroi
                    qdata = abs(FX_emp(:, n));
                    qmodel = abs(outFx(:,n));
                    qdata(isnan(qmodel)) = [];
                    qmodel(isnan(qmodel)) = [];
                    qdata = rescale(qdata);
                    qmodel = rescale(qmodel);
                    rvec(n) = corr(qdata, qmodel, 'type', 'Pearson', 'rows', 'complete');
                    errvec(n) = immse(qdata, qmodel);
                end
                rvec(isnan(rvec)) = 0;
        end
    end
end

% Forward model to predict fmri spectra:
function outFx = ForwardModel_FX(theta_star)
    % This is the forward model, that gives predicted fMRI freq spectrum from SC
    % Output: matrix of size nfft x nroi (in Fourier, not time)
    
    alpha = tanh(theta_star(1)); 
    tau = theta_star(2);     

    He = 1/tau^2./(1i*omegavec.'+1/tau).^2;
    frequency_response = ev_weight ./ bsxfun(@plus, 1i*omegavec.', 1/tau*(1 - alpha*(1 - ev)) *He );  % k x nt % new line
   
    % Define UtP (driving function) with varying size: k x 1 (for ones, randn) or 1 (for U) or k x nroi (for none)
    UtP = U' * ones(nroi,1);
    
    outFx = zeros(nfft, nroi);
    for n = 1:nfft
        outFx(n,:) =  (U * bsxfun(@times, frequency_response(:,n), UtP)).'; %new line
    end

end

end
