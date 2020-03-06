function [st_uv_KLT] = applyKLT_GPU(klt_s,klt_t,klt_u,klt_v,st_uv_single,N)
    st_uv_single = gpuArray(single(st_uv_single));
    
    N_st = length(klt_s);
    N_uv = length(klt_u);
    inv_klt_s = inv(klt_s);
    inv_klt_t = inv(klt_t);
    inv_klt_u = inv(klt_u);
    inv_klt_v = inv(klt_v);
    

    [Nt,Ns,Nv,Nu] = size(st_uv_single);
    
%-------------------------- 1-D KLT in t,s,v,u ---------------------------
    
    st_uv_KLT = permute(st_uv_single,[2 3 4 1]);
    st_uv_KLT = reshape(st_uv_KLT,N_st,1,[]);
    st_uv_KLT = pagefun(@mtimes,klt_t,st_uv_KLT);
    st_uv_KLT = reshape(st_uv_KLT,Ns,Nv,Nu,Nt);
    st_uv_KLT = permute(st_uv_KLT,[4 1 2 3]);
    
    st_uv_KLT = permute(st_uv_KLT,[1 3 4 2]);
    st_uv_KLT = reshape(st_uv_KLT,N_st,1,[]);
    st_uv_KLT = pagefun(@mtimes,klt_s,st_uv_KLT);
    st_uv_KLT = reshape(st_uv_KLT,Nt,Nv,Nu,Ns);
    st_uv_KLT = permute(st_uv_KLT,[1 4 2 3]);
    
    st_uv_KLT = permute(st_uv_KLT,[1 2 4 3]);
    st_uv_KLT = reshape(st_uv_KLT,N_uv,1,[]);
    st_uv_KLT = pagefun(@mtimes,klt_v,st_uv_KLT);
    st_uv_KLT = reshape(st_uv_KLT,Nt,Ns,Nu,Nv);
    st_uv_KLT = permute(st_uv_KLT,[1 2 4 3]);
    
    st_uv_KLT = reshape(st_uv_KLT,N_uv,1,[]);
    st_uv_KLT = pagefun(@mtimes,klt_u,st_uv_KLT);
    st_uv_KLT = reshape(st_uv_KLT,Nt,Ns,Nv,Nu);
    
%--------------------------- Remove Coefficients ------------------------
    
    st_uv_KLT = gather(st_uv_KLT);
    for nt = 1:N_st:Nt
        for ns = 1:N_st:Ns
            for nv = 1:N_uv:Nv
                for nu = 1:N_uv:Nu
                    blockLF = st_uv_KLT(nt:nt+N_st-1,ns:ns+N_st-1,nv:nv+N_uv-1,nu:nu+N_uv-1);
                    percent = prctile(abs(blockLF(:)),N);
                    ind = abs(blockLF) < percent;
                    blockLF(ind) = 0;
                    st_uv_KLT(nt:nt+N_st-1,ns:ns+N_st-1,nv:nv+N_uv-1,nu:nu+N_uv-1) = blockLF;
                end
            end
        end
    end
    st_uv_KLT = gpuArray(st_uv_KLT);

%     tempLF = zeros(size(st_uv_KLT),'single','gpuArray');
%     tempLF(1,1,:,:) = squeeze(st_uv_KLT(1,1,:,:));
%     tempLF(1,9,:,:) = squeeze(st_uv_KLT(1,9,:,:));
%     tempLF(9,1,:,:) = squeeze(st_uv_KLT(9,1,:,:));
%     tempLF(9,9,:,:) = squeeze(st_uv_KLT(9,9,:,:));
%     tempLF(2,1,:,:) = squeeze(st_uv_KLT(2,1,:,:));
%     tempLF(2,9,:,:) = squeeze(st_uv_KLT(2,9,:,:));
%     tempLF(10,1,:,:) = squeeze(st_uv_KLT(10,1,:,:));
%     tempLF(10,9,:,:) = squeeze(st_uv_KLT(10,9,:,:));
%     
%     st_uv_KLT = tempLF;
%     tempLF = [];

%     tempLF = zeros(size(st_uv_KLT),'single','gpuArray');
%     tempLF(1:8,1,:,:) = st_uv_KLT(1:8,1,:,:);
%     tempLF(1,1:8,:,:) = st_uv_KLT(1,1:8,:,:);
%     st_uv_KLT = tempLF;
%     tempLF = [];

%    nnz(st_uv_KLT)/numel(st_uv_KLT)
    
%----------------------- 1-D Inverse KLT in t,s,v,u -----------------------
    
    st_uv_KLT = reshape(st_uv_KLT,N_uv,1,[]);
    st_uv_KLT = pagefun(@mtimes,inv_klt_u,st_uv_KLT);
    st_uv_KLT = reshape(st_uv_KLT,Nt,Ns,Nv,Nu);
    
    st_uv_KLT = permute(st_uv_KLT,[1 2 4 3]);
    st_uv_KLT = reshape(st_uv_KLT,N_uv,1,[]);
    st_uv_KLT = pagefun(@mtimes,inv_klt_v,st_uv_KLT);
    st_uv_KLT = reshape(st_uv_KLT,Nt,Ns,Nu,Nv);
    st_uv_KLT = permute(st_uv_KLT,[1 2 4 3]);
    
    st_uv_KLT = permute(st_uv_KLT,[1 3 4 2]);
    st_uv_KLT = reshape(st_uv_KLT,N_st,1,[]);
    st_uv_KLT = pagefun(@mtimes,inv_klt_s,st_uv_KLT);
    st_uv_KLT = reshape(st_uv_KLT,Nt,Nv,Nu,Ns);
    st_uv_KLT = permute(st_uv_KLT,[1 4 2 3]);
    
    st_uv_KLT = permute(st_uv_KLT,[2 3 4 1]);
    st_uv_KLT = reshape(st_uv_KLT,N_st,1,[]);
    st_uv_KLT = pagefun(@mtimes,inv_klt_t,st_uv_KLT);
    st_uv_KLT = reshape(st_uv_KLT,Ns,Nv,Nu,Nt);
    st_uv_KLT = permute(st_uv_KLT,[4 1 2 3]); 
    
    st_uv_KLT = gather(uint16(st_uv_KLT));
    
end