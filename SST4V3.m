function opinionDynamicsGUI
    % Opinion Dynamics Simulator GUI
    fig = uifigure('Name','Opinion Dynamics Simulator','Position',[100 100 380 580]);

    %--- Network size ---
    uilabel(fig, 'Position',[20 530 120 22], 'Text','Network size:');
    ddNet = uidropdown(fig, ...
        'Position',[150 530 180 22], ...
        'Items',{'10×10','20×20','30×30','40×40','50×50'}, ...
        'ItemsData',[10,20,30,40,50], ...
        'Value',20, ...
        'Tooltip','Number of agents per row (grid is n×n)');

    %--- Neighborhood size ---
    uilabel(fig, 'Position',[20 490 120 22], 'Text','Neighborhood size:');
    ddNbr = uidropdown(fig, ...
        'Position',[150 490 180 22], ...
        'Items',{'3×3','5×5','7×7','9×9'}, ...
        'ItemsData',[3,5,7,9], ...
        'Value',5, ...
        'Tooltip','Side length of local Moore neighborhood (odd)');

    %--- w1 weight ---
    uilabel(fig,'Position',[20 450 120 22],'Text','w1 (0–1):');
    fldW1 = uieditfield(fig,'numeric','Position',[150 450 180 22],...
                        'Limits',[0 1],'Value',0.5, ...
                        'Tooltip','Weight on authenticity concern');

    %--- UL upper bound ---
    uilabel(fig,'Position',[20 410 120 22],'Text','Norm certainty (≥10):');
    fldUL = uieditfield(fig,'numeric','Position',[150 410 180 22],...
                        'Limits',[10 Inf],'Value',50, ...
                        'Tooltip','Max total pseudocount for social norms');

    %--- gamma sensitivity ---
    uilabel(fig,'Position',[20 370 120 22],'Text','gamma (≥1):');
    fldG = uieditfield(fig,'numeric','Position',[150 370 180 22],...
                       'Limits',[1 Inf],'Value',10, ...
                       'Tooltip','Sensitivity of disutility function');

    %--- sumage certainty ---
    uilabel(fig,'Position',[20 330 120 22],'Text','Aut certainty (≥10):');
    fldS = uieditfield(fig,'numeric','Position',[150 330 180 22],...
                       'Limits',[10 Inf],'Value',20, ...
                       'Tooltip','Certainty weight for authentic preference');

    %--- mutation rate ---
    uilabel(fig,'Position',[20 290 120 22],'Text','mutation rate (0–0.5):');
    fldM = uieditfield(fig,'numeric','Position',[150 290 180 22],...
                       'Limits',[0 0.5],'Value',0.5, ...
                       'Tooltip','Initial group mis-assignment rate');

    %--- pActive fraction ---
    uilabel(fig,'Position',[20 250 120 22],'Text','pActive (0.01–1):');
    fldP = uieditfield(fig,'numeric','Position',[150 250 180 22],...
                       'Limits',[0.01 1],'Value',0.1, ...
                       'Tooltip','Fraction of agents activated each step');

    %--- RNG seed ---
    uilabel(fig,'Position',[20 210 120 22],'Text','Iteration (>1):');
    fldI = uieditfield(fig,'numeric','Position',[150 210 180 22],...
                           'Limits',[1 100000], 'Value',200, ...
                           'Tooltip','Number of iterations');

    %--- RNG seed ---
    uilabel(fig,'Position',[20 170 120 22],'Text','Seed (00000–99999):');
    fldSeed = uieditfield(fig,'numeric','Position',[150 170 180 22],...
                           'Limits',[0 99999], 'RoundFractionalValues',true, 'Value',1, ...
                           'Tooltip','Random seed for reproducibility');

    %--- Run button ---
    uibutton(fig, 'Position',[140 30 100 30], 'Text','Run Simulation', ...
        'ButtonPushedFcn', @(btn,event) runSimulation( ...
            ddNet.Value, ddNbr.Value, fldW1.Value, fldUL.Value, ...
            fldG.Value, fldS.Value, fldM.Value, fldP.Value, fldI.Value,fldSeed.Value) );
end

function runSimulation(nsize, neighborhoodSize, w1, UL, gamma, sumage, mutation, pActive,iteration,seed)
    % Compute derived
    w2 = 1 - w1;
    T  = iteration;
    halfWin = floor(neighborhoodSize/2);
    Ntot = nsize^2;
    
    % Initialize attitudes & groups…
    rng(seed);
    alpha1 = 1.001; beta1 = sumage - alpha1;
    lim1 = betainv(0.5,alpha1,beta1);
    lim2 = 1 - lim1;
    gap = (lim2-lim1)/(Ntot-1);
    initC = reshape(linspace(lim1,lim2,Ntot),nsize,nsize);
    perm = randperm(Ntot);
    initialchoice = reshape(initC(perm), nsize, nsize);
    % … and fit each agent’s alpha/beta to reproduce that median:
    agealpha = zeros(nsize); agebeta = zeros(nsize);
    for i=1:nsize, for j=1:nsize
        fun = @(x)abs(betainv(0.5,x,sumage-x)-initialchoice(i,j));
        agealpha(i,j) = fminsearchbnd(fun,1.1,1.001,sumage-1.001);
        agebeta(i,j)  = sumage - agealpha(i,j);
    end, end
    
    choice = initialchoice;
    group  = double(choice<0.5);
    % apply mutation flips…
    idx_low   = find(choice < 0.5);
    n_low     = numel(idx_low);
    n_flip_lo = round(mutation * n_low);
    flip_low  = idx_low(randperm(n_low, n_flip_lo));
    group(flip_low) = 0;

    idx_high   = find(choice >= 0.5);
    n_high     = numel(idx_high);
    n_flip_hi  = round(mutation * n_high);
    flip_high  = idx_high(randperm(n_high, n_flip_hi));
    group(flip_high) = 1;

    threat  = ones(nsize, nsize);
    utility = ones(nsize, nsize);
    
    % Prepare histories
    attHist   = zeros(nsize,nsize,T);
    groupHist = zeros(nsize,nsize,T);
    bins      = 0:0.01:1;
    distHist  = zeros(numel(bins)-1, T);
    threatAvg = zeros(1,T);
    
    % Build colormap
    ncol = 256;
    c1 = [linspace(0,1,ncol)', linspace(0,1,ncol)', ones(ncol,1)];
    c2 = [ones(ncol,1), linspace(1,0,ncol)', linspace(1,0,ncol)'];
    bwr = [c1; c2];
    
    % Create the simulation figure with 4 panels
    fig2 = figure('Name','Simulation','Position',[600 100 900 600]);
    ax1 = subplot(2,2,1,'Parent',fig2);
      hAtt  = imagesc(ax1, zeros(nsize)); colormap(ax1,bwr);
      title(ax1,'Attitudes'); axis(ax1,'square'); caxis([0 1]);
    ax2 = subplot(2,2,2,'Parent',fig2);
      hG1 = plot(ax2,NaN,NaN,'bo','MarkerFaceColor','b'); hold(ax2,'on');
      hG0 = plot(ax2,NaN,NaN,'kx'); set(ax2,'YDir','reverse');
      axis(ax2,[1 nsize 1 nsize],'square'); grid(ax2,'on');
      title(ax2,'Groups');
    ax3 = subplot(2,2,3,'Parent',fig2);
      [X,Y] = meshgrid(1:T,(bins(1:end-1)+bins(2:end))/2);
      hSurf = surf(ax3,X,Y,zeros(size(X)),'EdgeColor','none');
      view(ax3,2); colorbar(ax3);
      xlabel(ax3,'Iter'); ylabel(ax3,'Att value');
      title(ax3,'Distribution');
    ax4 = subplot(2,2,4,'Parent',fig2);
      hLine = plot(ax4,1,0,'LineWidth',1.5);
      axis(ax4,[1 T 0  8]);
      xlabel(ax4,'Iter'); ylabel(ax4,'Mean threat');
      title(ax4,'Threat');
    
    % Main loop
    for t = 1:T
        %% 1) randomly activate pActive of all agents
  nActive = round(pActive * Ntot);
  activeLin = randperm(Ntot, nActive);    % linear indices of activated agents

  %% 2) compute each activated agent’s current utility
  for k = 1:nActive
    idx = activeLin(k);
    [i,j] = ind2sub([nsize,nsize], idx);

    % get toroidal neighborhood rows & cols
    rows = mod((i-halfWin:i+halfWin)-1,nsize)+1;
    cols = mod((j-halfWin:j+halfWin)-1,nsize)+1;
    [RR,CC] = meshgrid(rows,cols);
    linBlock = sub2ind([nsize,nsize], RR(:), CC(:));
    linBlock( RR(:)==i & CC(:)==j ) = [];  % drop self

    % classify neighbors using group identity
    groupidentity = group(linBlock);
    neighChoice = choice(linBlock);
    group1sample = linBlock(find(groupidentity==1));
    group2sample = linBlock(find(groupidentity==0));
    group1choice = choice(group1sample);
    group2choice = choice(group2sample);

    % estimate soical norms from group1 and group2, represented by two beta
    % distributions
    % control the kurtosis of social norm distrbution expalpha+expbeta<UL


    [expalpha1,expbeta1] = fitBeta(group1choice, UL, sumage)
    [expalpha2,expbeta2] = fitBeta(group2choice, UL, sumage)


    % calculate utilities when the activated agent receives group1 and
    % group2 social norms

    x=0:0.001:1;

    y0=utilitya(x,agealpha(idx),agebeta(idx),gamma); % authencity-related utility
    y1=utilitya(x,expalpha1,expbeta1,gamma); % social conformity utility in group1
    y2=utilitya(x,expalpha2,expbeta2,gamma); % social conformity utility in group2

    yg1 = w1*y0+w2*y1;
    yg2 = w1*y0+w2*y2;
    [ystar1,xstar1] = min(yg1);
    [ystar2,xstar2] = min(yg2);

    if ystar1<=ystar2 % the agent choose change its new group identity to the group with lower disutility
        group(idx) = 1;
        choice(idx) = x(xstar1);
        outgroupsize = length(group2sample);
        ingroupU = ystar1;
        outgroupU = ystar2;
    else
        group(idx) = 0;
        choice(idx) = x(xstar2);
        outgroupsize = length(group1sample);
        ingroupU = ystar2;
        outgroupU = ystar1;
    end

    % calculate the perceived threat from outgroup
    % element1: probability of outgroup wins = outgroupsize/neighoursize
    loseprob = outgroupsize/length(linBlock);
    % element2: utility loss when outgroup wins
    relativeloss = outgroupU/ingroupU;
    % perceived threat = element1*element2
    threat(idx) = loseprob*relativeloss;
    utility(idx) = ingroupU/outgroupU;
  end

  %% 3) random pairing of activated agents
  perm = randperm(nActive);
  if mod(nActive,2)==1
    perm(end) = []; % drop one if odd
    nPairs = (nActive-1)/2;
  else
    nPairs = nActive/2;
  end

  %% 4) for each pair, swap if both receive high outgroup threat
  for p = 1:nPairs
    a = activeLin(perm(2*p-1));
    b = activeLin(perm(2*p));

    % current (row,col)
    [ia,ja] = ind2sub([nsize,nsize], a);
    [ib,jb] = ind2sub([nsize,nsize], b);

    % old threats
    T_a_old = threat(a);
    T_b_old = threat(b);

    % “what if” at each other’s location?
    T_a_new = computeThreatAt(a, ib, jb, choice, group, agealpha, agebeta, w1, w2, gamma, UL, halfWin, nsize, sumage);
    T_b_new = computeThreatAt(b, ia, ja, choice, group, agealpha, agebeta, w1, w2, gamma, UL, halfWin, nsize, sumage);

    if T_a_new < T_a_old && T_b_new < T_b_old
      % swap ALL state fields, and update to the new threats
      tmp.choice    = choice(ia,ja);
      tmp.group     = group(ia,ja);
      tmp.alpha     = agealpha(ia,ja);
      tmp.beta      = agebeta(ia,ja);
      tmp.util      = utility(ia,ja);
      tmp.threat    = threat(ia,ja);
      tmp.int       = initialchoice(ia,ja);

      % A ← B’s state
      choice(ia,ja)   = choice(ib,jb);
      group(ia,ja)    = group(ib,jb);
      agealpha(ia,ja) = agealpha(ib,jb);
      agebeta(ia,ja)  = agebeta(ib,jb);
      utility(ia,ja)  = utility(ib,jb);
      threat(ia,ja)   = T_a_new;       
      initialchoice(ia,ja) = initialchoice(ib,jb);

      % B ← A’s old state
      choice(ib,jb)   = tmp.choice;
      group(ib,jb)    = tmp.group;
      agealpha(ib,jb) = tmp.alpha;
      agebeta(ib,jb)  = tmp.beta;
      utility(ib,jb)  = tmp.util;
      threat(ib,jb)   = T_b_new;
      initialchoice(ib,jb) = tmp.int;
    end
  end
      % at end of step:
      attHist(:,:,t)   = choice;
      groupHist(:,:,t) = group;
      distHist(:,t)    = histcounts(choice(:),bins,'Normalization','probability');
      threatAvg(t)     = mean(threat(:));
      
      % 1) update attitude heatmap
      set(hAtt,'CData', attHist(:,:,t));
      % 2) update group scatter
      [i1,j1] = find(groupHist(:,:,t)==1);
      [i0,j0] = find(groupHist(:,:,t)==0);
      set(hG1,'XData',j1,'YData',i1);
      set(hG0,'XData',j0,'YData',i0);
      % 3) update distribution surface
      set(hSurf,'ZData', distHist);
      % 4) update threat curve
      set(hLine,'XData',1:t,'YData',threatAvg(1:t));
      
      drawnow;
      pause(0.01);
    end
end

%% Utility function
% note that it called utility function, but it acturally means 'disutility'
% the opitmal choice means the one minimize this function
function disutility=utilitya(x,alpha1,beta1,gamma)
I1=betacdf(x,alpha1,beta1);
disutility=exp(abs(gamma*(I1-0.5)));
end

%% Threat function
function T = computeThreatAt(agentIdx, row, col, choice, group, agealpha, agebeta, w1, w2, gamma, UL, halfWin, nsize, sumage)
  % uses parent workspace variables:
  %   choice, group, agealpha, agebeta, w1, w2, gamma, UL, halfWin, nsize, sumage

  % 1) gather the Moore neighborhood (toroidal wrap)
  rows = mod((row-halfWin : row+halfWin)-1, nsize) + 1;
  cols = mod((col-halfWin : col+halfWin)-1, nsize) + 1;
  [RR,CC] = meshgrid(rows, cols);
  blockIdx = sub2ind([nsize,nsize], RR(:), CC(:));
  blockIdx( RR(:)==row & CC(:)==col ) = [];   % drop self

  % 2) split into the two groups
  neighChoices = choice(blockIdx);
  neighGroup   = group(blockIdx);
  G1 = neighChoices(neighGroup==1);
  G0 = neighChoices(neighGroup==0);

  % 3) fit beta‐params for each (with your UL truncation)
  [a1,b1] = fitBeta(G1, UL, sumage);
  [a2,b2] = fitBeta(G0, UL, sumage);

  % 4) compute utilities over x∈[0,1]
  x  = 0:0.001:1;
  y0 = utilitya(x, agealpha(agentIdx), agebeta(agentIdx), gamma);
  y1 = utilitya(x, a1, b1, gamma);
  y2 = utilitya(x, a2, b2, gamma);

  % 5) ingroup vs outgroup utility for THIS agent’s current group
  if group(agentIdx)==1
    Uin  = min(w1*y0 + w2*y1);
    Uout = min(w1*y0 + w2*y2);
    nOut = numel(G0);
  else
    Uin  = min(w1*y0 + w2*y2);
    Uout = min(w1*y0 + w2*y1);
    nOut = numel(G1);
  end

  % 6) perceived threat
  loseProb = nOut / numel(blockIdx);
  T = loseProb * (Uout / Uin);
end

%% Social norm estimate function
function [alphaHat,betaHat] = fitBeta(data, UL, sumage)
  if numel(data) >= 2
    mu = mean(data); sd = std(data);
    alphaHat = ((1-mu)*mu^2)/(sd^2) - mu;
    betaHat  = ((1-mu)/mu)*alphaHat;
  else
    alphaHat = 2; betaHat = 2;
  end
  % enforce UL max, and a,b ≥1.001
  s = alphaHat + betaHat;
  if s > UL
    alphaHat = UL*(alphaHat/s);  alphaHat = max(alphaHat,1.001);
    betaHat  = UL*(betaHat/s);   betaHat  = max(betaHat,1.001);
  end
end

%% fminsearchbnd
function [x,fval,exitflag,output] = fminsearchbnd(fun,x0,LB,UB,options,varargin)

% Example usage:
% rosen = @(x) (1-x(1)).^2 + 105*(x(2)-x(1).^2).^2;
%
% fminsearch(rosen,[3 3])     % unconstrained
% ans =
%    1.0000    1.0000
%
% fminsearchbnd(rosen,[3 3],[2 2],[])     % constrained
% ans =
%    2.0000    4.0000
%
% See test_main.m for other examples of use.

xsize = size(x0);
x0 = x0(:);
n=length(x0);

if (nargin<3) || isempty(LB)
  LB = repmat(-inf,n,1);
else
  LB = LB(:);
end
if (nargin<4) || isempty(UB)
  UB = repmat(inf,n,1);
else
  UB = UB(:);
end

if (n~=length(LB)) || (n~=length(UB))
  error 'x0 is incompatible in size with either LB or UB.'
end

% set default options if necessary
if (nargin<5) || isempty(options)
  options = optimset('fminsearch');
end

% stuff into a struct to pass around
params.args = varargin;
params.LB = LB;
params.UB = UB;
params.fun = fun;
params.n = n;
% note that the number of parameters may actually vary if 
% a user has chosen to fix one or more parameters
params.xsize = xsize;
params.OutputFcn = [];

% 0 --> unconstrained variable
% 1 --> lower bound only
% 2 --> upper bound only
% 3 --> dual finite bounds
% 4 --> fixed variable
params.BoundClass = zeros(n,1);
for i=1:n
  k = isfinite(LB(i)) + 2*isfinite(UB(i));
  params.BoundClass(i) = k;
  if (k==3) && (LB(i)==UB(i))
    params.BoundClass(i) = 4;
  end
end

% transform starting values into their unconstrained
% surrogates. Check for infeasible starting guesses.
x0u = x0;
k=1;
for i = 1:n
  switch params.BoundClass(i)
    case 1
      % lower bound only
      if x0(i)<=LB(i)
        % infeasible starting value. Use bound.
        x0u(k) = 0;
      else
        x0u(k) = sqrt(x0(i) - LB(i));
      end
      
      % increment k
      k=k+1;
    case 2
      % upper bound only
      if x0(i)>=UB(i)
        % infeasible starting value. use bound.
        x0u(k) = 0;
      else
        x0u(k) = sqrt(UB(i) - x0(i));
      end
      
      % increment k
      k=k+1;
    case 3
      % lower and upper bounds
      if x0(i)<=LB(i)
        % infeasible starting value
        x0u(k) = -pi/2;
      elseif x0(i)>=UB(i)
        % infeasible starting value
        x0u(k) = pi/2;
      else
        x0u(k) = 2*(x0(i) - LB(i))/(UB(i)-LB(i)) - 1;
        % shift by 2*pi to avoid problems at zero in fminsearch
        % otherwise, the initial simplex is vanishingly small
        x0u(k) = 2*pi+asin(max(-1,min(1,x0u(k))));
      end
      
      % increment k
      k=k+1;
    case 0
      % unconstrained variable. x0u(i) is set.
      x0u(k) = x0(i);
      
      % increment k
      k=k+1;
    case 4
      % fixed variable. drop it before fminsearch sees it.
      % k is not incremented for this variable.
  end
  
end
% if any of the unknowns were fixed, then we need to shorten
% x0u now.
if k<=n
  x0u(k:n) = [];
end

% were all the variables fixed?
if isempty(x0u)
  % All variables were fixed. quit immediately, setting the
  % appropriate parameters, then return.
  
  % undo the variable transformations into the original space
  x = xtransform(x0u,params);
  
  % final reshape
  x = reshape(x,xsize);
  
  % stuff fval with the final value
  fval = feval(params.fun,x,params.args{:});
  
  % fminsearchbnd was not called
  exitflag = 0;
  
  output.iterations = 0;
  output.funcCount = 1;
  output.algorithm = 'fminsearch';
  output.message = 'All variables were held fixed by the applied bounds';
  
  % return with no call at all to fminsearch
  return
end

% Check for an outputfcn. If there is any, then substitute my
% own wrapper function.
if ~isempty(options.OutputFcn)
  params.OutputFcn = options.OutputFcn;
  options.OutputFcn = @outfun_wrapper;
end

% now we can call fminsearch, but with our own
% intra-objective function.
[xu,fval,exitflag,output] = fminsearch(@intrafun,x0u,options,params);

% undo the variable transformations into the original space
x = xtransform(xu,params);

% final reshape to make sure the result has the proper shape
x = reshape(x,xsize);

% Use a nested function as the OutputFcn wrapper
  function stop = outfun_wrapper(x,varargin);
    % we need to transform x first
    xtrans = xtransform(x,params);
    
    % then call the user supplied OutputFcn
    stop = params.OutputFcn(xtrans,varargin{1:(end-1)});
    
  end

end % mainline end

% ======================================
% ========= begin subfunctions =========
% ======================================
function fval = intrafun(x,params)
% transform variables, then call original function

% transform
xtrans = xtransform(x,params);

% and call fun
fval = feval(params.fun,reshape(xtrans,params.xsize),params.args{:});

end % sub function intrafun end

% ======================================
function xtrans = xtransform(x,params)
% converts unconstrained variables into their original domains

xtrans = zeros(params.xsize);
% k allows some variables to be fixed, thus dropped from the
% optimization.
k=1;
for i = 1:params.n
  switch params.BoundClass(i)
    case 1
      % lower bound only
      xtrans(i) = params.LB(i) + x(k).^2;
      
      k=k+1;
    case 2
      % upper bound only
      xtrans(i) = params.UB(i) - x(k).^2;
      
      k=k+1;
    case 3
      % lower and upper bounds
      xtrans(i) = (sin(x(k))+1)/2;
      xtrans(i) = xtrans(i)*(params.UB(i) - params.LB(i)) + params.LB(i);
      % just in case of any floating point problems
      xtrans(i) = max(params.LB(i),min(params.UB(i),xtrans(i)));
      
      k=k+1;
    case 4
      % fixed variable, bounds are equal, set it at either bound
      xtrans(i) = params.LB(i);
    case 0
      % unconstrained variable.
      xtrans(i) = x(k);
      
      k=k+1;
  end
end

end % sub function xtransform end