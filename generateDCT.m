function [dct_st, adct_st] = generateDCT(N)
    dct_st = dctmtx(N);
    adct_st = sign(dct_st);
end