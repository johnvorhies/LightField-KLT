N = [4 8 16];
cRate = [0:2:18 20:5:45 50:10:95];
cRate(1) = 1;

[Nt,Ns,Nv,Nu] = size(st_uv);

PSNR_exact = zeros(9,length(cRate),'single','gpuArray');
SSIM_exact = zeros(9,length(cRate),'single','gpuArray');
PSNR_approx = zeros(9,length(cRate),'single','gpuArray');
SSIM_approx = zeros(9,length(cRate),'single','gpuArray');

% Tarot
rho_s = 0.8444;
rho_t = 0.8928;
rho_u = 0.8442;
rho_v = 0.8442;

%------------------ Exact KLT --------------------------
tic
for n = 1:length(N)
    [klt_s,sklt_s] = generateKLT(rho_s,N(n));
    [klt_t,sklt_t] = generateKLT(rho_t,N(n));
    [klt_u,sklt_u] = generateKLT(rho_u,N(n));
    [klt_v,sklt_v] = generateKLT(rho_v,N(n));
    
    klt_s = gpuArray(single(klt_s));
    klt_t = gpuArray(single(klt_t));
    klt_u = gpuArray(single(klt_u));
    klt_v = gpuArray(single(klt_v));
   
    parfor c = 1:length(cRate)
        [st_uv_klt] = applyKLT_GPU(klt_s,klt_t,klt_u,klt_v,st_uv,cRate(c));
        PSNR_exact(n,c) = psnr(st_uv_klt,st_uv);
        SSIM_exact(n,c) = ssim(squeeze(st_uv_klt(8,8,:,:)),squeeze(st_uv(8,8,:,:)));
        st_uv_klt = [];
    end
    poolobj = gcp('nocreate');
    delete(poolobj);
end

for n = 1:length(N)
    [klt_s,sklt_s] = generateKLT(rho_s,N(n));
    [klt_t,sklt_t] = generateKLT(rho_t,N(n));
    [klt_u,sklt_u] = generateKLT(rho_u,32);
    [klt_v,sklt_v] = generateKLT(rho_v,32);
    
    klt_s = gpuArray(single(klt_s));
    klt_t = gpuArray(single(klt_t));
    klt_u = gpuArray(single(klt_u));
    klt_v = gpuArray(single(klt_v));
    
    ind = n+3;
    
    parfor c = 1:length(cRate)
        [st_uv_klt] = applyKLT_GPU(klt_s,klt_t,klt_u,klt_v,st_uv,cRate(c));
        PSNR_exact(ind,c) = psnr(st_uv_klt,st_uv);
        SSIM_exact(ind,c) = ssim(squeeze(st_uv_klt(8,8,:,:)),squeeze(st_uv(8,8,:,:)));
        st_uv_klt = [];
    end
    poolobj = gcp('nocreate');
    delete(poolobj);
end

for n = 1:length(N)
    [klt_s,sklt_s] = generateKLT(rho_s,N(n));
    [klt_t,sklt_t] = generateKLT(rho_t,N(n));
    [klt_u,sklt_u] = generateKLT(rho_u,64);
    [klt_v,sklt_v] = generateKLT(rho_v,64);
    
    klt_s = gpuArray(single(klt_s));
    klt_t = gpuArray(single(klt_t));
    klt_u = gpuArray(single(klt_u));
    klt_v = gpuArray(single(klt_v));
    
    ind = n+6;
    
    parfor c = 1:length(cRate)
        [st_uv_klt] = applyKLT_GPU(klt_s,klt_t,klt_u,klt_v,st_uv,cRate(c));
        PSNR_exact(ind,c) = psnr(st_uv_klt,st_uv);
        SSIM_exact(ind,c) = ssim(squeeze(st_uv_klt(8,8,:,:)),squeeze(st_uv(8,8,:,:)));
        st_uv_klt = [];
    end
    poolobj = gcp('nocreate');
    delete(poolobj);
end

%-------------------- SKLT ---------------------------------

for n = 1:length(N)
    [klt_s,sklt_s] = generateKLT(rho_s,N(n));
    [klt_t,sklt_t] = generateKLT(rho_t,N(n));
    [klt_u,sklt_u] = generateKLT(rho_u,N(n));
    [klt_v,sklt_v] = generateKLT(rho_v,N(n));
    
    sklt_s = gpuArray(single(sklt_s));
    sklt_t = gpuArray(single(sklt_t));
    sklt_u = gpuArray(single(sklt_u));
    sklt_v = gpuArray(single(sklt_v));
   
    parfor c = 1:length(cRate)
        [st_uv_klt] = applyKLT_GPU(sklt_s,sklt_t,sklt_u,sklt_v,st_uv,cRate(c));
        PSNR_approx(n,c) = psnr(st_uv_klt,st_uv);
        SSIM_approx(n,c) = ssim(squeeze(st_uv_klt(8,8,:,:)),squeeze(st_uv(8,8,:,:)));
        st_uv_klt = [];
    end
    poolobj = gcp('nocreate');
    delete(poolobj);
end

for n = 1:length(N)
    [klt_s,sklt_s] = generateKLT(rho_s,N(n));
    [klt_t,sklt_t] = generateKLT(rho_t,N(n));
    [klt_u,sklt_u] = generateKLT(rho_u,32);
    [klt_v,sklt_v] = generateKLT(rho_v,32);
    
    sklt_s = gpuArray(single(sklt_s));
    sklt_t = gpuArray(single(sklt_t));
    sklt_u = gpuArray(single(sklt_u));
    sklt_v = gpuArray(single(sklt_v));
    
    ind = n+3;
    
    parfor c = 1:length(cRate)
        [st_uv_klt] = applyKLT_GPU(sklt_s,sklt_t,sklt_u,sklt_v,st_uv,cRate(c));
        PSNR_approx(ind,c) = psnr(st_uv_klt,st_uv);
        SSIM_approx(ind,c) = ssim(squeeze(st_uv_klt(8,8,:,:)),squeeze(st_uv(8,8,:,:)));
        st_uv_klt = [];
    end
    poolobj = gcp('nocreate');
    delete(poolobj);
end

for n = 1:length(N)
    [klt_s,sklt_s] = generateKLT(rho_s,N(n));
    [klt_t,sklt_t] = generateKLT(rho_t,N(n));
    [klt_u,sklt_u] = generateKLT(rho_u,64);
    [klt_v,sklt_v] = generateKLT(rho_v,64);
    
    sklt_s = gpuArray(single(sklt_s));
    sklt_t = gpuArray(single(sklt_t));
    sklt_u = gpuArray(single(sklt_u));
    sklt_v = gpuArray(single(sklt_v));
    
    ind = n+6;
    
    parfor c = 1:length(cRate)
        [st_uv_klt] = applyKLT_GPU(sklt_s,sklt_t,sklt_u,sklt_v,st_uv,cRate(c));
        PSNR_approx(ind,c) = psnr(st_uv_klt,st_uv);
        SSIM_approx(ind,c) = ssim(squeeze(st_uv_klt(8,8,:,:)),squeeze(st_uv(8,8,:,:)));
        st_uv_klt = [];
    end
    poolobj = gcp('nocreate');
    delete(poolobj);
end

PSNR_exact = gather(PSNR_exact);
PSNR_approx = gather(PSNR_approx);
SSIM_exact = gather(SSIM_exact);
SSIM_approx = gather(SSIM_approx);
toc

