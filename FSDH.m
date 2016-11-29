function [B, R] = FSDH(X, y, bit)
    HA = hadamard(bit);
    B  = HA(y,:);
    R  = (X'*X)\(X'*B);
end