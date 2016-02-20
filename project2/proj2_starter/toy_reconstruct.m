function toy_reconstruct(toyim)
%toy_reconstruct(toyim)
%reconstrut image from gradient
[lh,lw] = size(toyim);

kPix = lh * lw;
kEdge = (lh-1) * (lw-1);
im2var = zeros(lh,lw);

im2var(1:kPix) = 1:kPix;

triplet = zeros(kEdge*3+1, 3);

%fill sparse matrix by triplet
for x=1:lw-1
    for y=1:lh-1
        triplet(im2var(y,x)
    end
end
end