function [dct_st, adct_st] = generateDCT(N)
    dct_st = dctmtx(N);
    adct_st = 1/sqrt(N) * sign(dct_st);
end