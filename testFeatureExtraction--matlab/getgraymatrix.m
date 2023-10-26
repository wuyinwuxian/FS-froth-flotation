function gm = getgraymatrix(Frame,graylevel,d,a,t)
    neighbornums = (2*d+1)^2;                  % 邻域个数
    [px,py] = getgrayfrequency(Frame,d,a,t);   %得到灰度频次表
    [row,col] = size(px);
    gm = zeros(graylevel,neighbornums);
    for i = 1 : row
        for j = 1 : col
            gm(px(i,j)+1,py(i,j)+1) = gm(px(i,j)+1,py(i,j)+1)+1;
        end
    end
end