function [fs, bs, s] = forward_backward(Y,alpha,eta)
Nstate = size(alpha,1);
Nout = size(eta,2);
Y = [Nout+1, Y ];
Nt = length(Y);
fs = zeros(Nstate,Nt);
fs(1,1) = 1;  
s = zeros(1,Nt);
s(1) = 1;
for count = 2:Nt
    for state = 1:Nstate
        fs(state,count) = eta(state,Y(count),count-1) .* (sum(fs(:,count-1) .*alpha(:,state,count-1)));
        if isnan(fs(state,count))
            error('NaN in fs');
        end
    end
    s(count) =  sum(fs(:,count));
    if isnan(s(count))
            error('NaN in s');
    end
    fs(:,count) =  fs(:,count)./s(count);

end
f = fs.*repmat(cumprod(s),size(fs,1),1);

bs = ones(Nstate,Nt);
for count = Nt-1:-1:1
    for state = 1:Nstate
      bs(state,count) = (1/s(count+1)) * sum( alpha(state,:,count)'.* bs(:,count+1) .* eta(:,Y(count+1),count)); 
      if isnan(bs(state,count))
            error('NaN in bs');
        end
    end
end
scales = cumprod(s, 'reverse'); 
b = bs.*repmat([scales(2:end), 1],size(bs,1),1);
