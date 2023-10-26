function gm = getgraymatrix(Frame,graylevel,d,a,t)
    neighbornums = (2*d+1)^2;                  % �������
    [px,py] = getgrayfrequency(Frame,d,a,t);   %�õ��Ҷ�Ƶ�α�
    [row,col] = size(px);
    gm = zeros(graylevel,neighbornums);
    for i = 1 : row
        for j = 1 : col
            gm(px(i,j)+1,py(i,j)+1) = gm(px(i,j)+1,py(i,j)+1)+1;
        end
    end
end