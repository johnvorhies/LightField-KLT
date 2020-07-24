function [KLT,aKLT] = generateKLT(rho,N)
    % Generate covariance matrix
    K = zeros(N,N);
    for i = 1:N
        for j = 1:N
            K(i,j) = rho^abs(i-j);
        end
    end
    
    lambda = eig(K);
    lambda = sort(lambda,'descend');
    omega = acos((1-rho^2-lambda-lambda*rho^2)./(-2*lambda*rho));
    
    KLT = zeros(N,N);
    for i = 0:N-1
        for j = 0:N-1
            w = sqrt(2/(N+lambda(i+1)))*sin(omega(i+1)*((j+1)-(N+1)/2)+...
                ((i+1)*pi)/2);
            KLT(i+1,j+1) = w;
        end
    end
    
    aKLT = sign(KLT);
end