
function [K] = compute_rbf_kernel(x,XA,y,sqrsig)
    xmXA = bsxfun(@minus, x, XA);
    K = exp(-sum(xmXA.^2,1)/(2*sqrsig));
end
