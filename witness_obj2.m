function [loss,grad] = witness_obj2(r,x,XA,XB,sqrsig,lambda)

    na = size(XA,2);
    nb = size(XB,2);
    KA = compute_rbf_kernel(x + r,XA,0,sqrsig); % (1,na)

    KB = compute_rbf_kernel(x + r,XB,0,sqrsig);
    loss = (1/na)*sum(KA) - (1/nb)*sum(KB);
    
    loss = loss + lambda*(r'*r);
    loss
    
    
    xmXA = bsxfun(@minus, x + r, XA); % (d,na)
    xmXB = bsxfun(@minus, x + r, XB); % (d,nb)
    

    GA = (-1/(na*sqrsig))*sum(bsxfun(@times, KA, xmXA),2);
    
    GB = (-1/(nb*sqrsig))*sum(bsxfun(@times, KB, xmXB),2);
    grad = GA - GB;
    grad = grad + 2*lambda*r;
    [min(grad(:)) max(grad(:))] % debug
    %keyboard
end 
