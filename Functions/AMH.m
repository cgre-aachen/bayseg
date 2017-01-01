function [samples,acceptRate,coviance]=AMH(target,targetArgs,x0,Nsamples,C0)
% Adaptive Metropolis-Hastings algorithm
%
% Inputs
% target returns the unnormalized log posterior, called as p = exp(target(x, targetArgs{:}))
% targetArgs - cell array passed to target?
% x0 is a 1xd vector! specifying the initial state
% Nsamples - total number of samples to draw
% SIGMA - the initial standard diviation of all dimentions for proposal function.
%
% Outputs
% samples(s,:) is the s’th sample (of size d)
% acceptRate = percentage of accepted moves

    function xprime=GaussianProposal(x,SigmaProp)
        % only generate one candidate within a sigma_proposal (C)
        xprime=mvnrnd(x,SigmaProp);
    end

d = length(x0);
samples = zeros(Nsamples,d);

t0=100;
error=1e-5;
sd=2.4^2/d;

x = x0;
x_bar=x;
C=C0;
naccept = 0;
logpOld = feval(target,x,targetArgs);

for t=1:Nsamples
    x_barBefore=x_bar;
    xprime = GaussianProposal(x,C);
    logpNew = feval(target,xprime,targetArgs);
    alpha = exp(logpNew - logpOld);
    r = min(1, alpha);
    u = rand(1);
    if u < r
        x = xprime;
        naccept = naccept + 1;
        logpOld = logpNew;
    end
    samples(t,:) = x;
    x_bar=(x_barBefore*t+x)/(t+1);
    if t>=t0+1
        C=(t-1)/t*C+sd/t*(t*(x_barBefore'*x_barBefore)-(t+1)*(x_bar'*x_bar)+(x'*x)+error*eye(d));
    end
end
acceptRate=naccept/Nsamples;
coviance=C;
end
