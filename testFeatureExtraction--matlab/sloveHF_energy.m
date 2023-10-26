function HF_energy = sloveHF_energy(im,graylevel,d,a,t)
    ngldm = getgraymatrix(im,graylevel,d,a,t);
    [row,col] = size(ngldm);
    tempSum = 0;
    for i = 1 : row
        for j = 1 : col
            tempSum = tempSum + i*double(ngldm(i , j).^2);
        end
    end
    HF_energy = tempSum;       %18 ¸ßÆµÄÜÁ¿
end