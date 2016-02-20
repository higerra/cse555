function toy_reconstruct(toyim)
%toy_reconstruct(toyim)
%reconstrut image from gradient
[lh,lw] = size(toyim);

kPix = lh * lw;
kEdge = (lh-1) * (lw-1);
im2var = zeros(lh,lw);

im2var(1:kPix) = 1:kPix;

triplet = zeros(kEdge*2+1, 3);

%fill sparse matrix by triplet
e = 1;
for y=1:lh-1
   for x=1:lw-1
       triplet(e:e+1,1) = [im2var(y,x), im2var(y,x)];
       
   end
end

end