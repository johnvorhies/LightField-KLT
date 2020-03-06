function [DCT, SDCT] = generateDCT(N)
    DCT = dctmtx(N);
    SDCT = 1/sqrt(N) * sign(DCT);
end