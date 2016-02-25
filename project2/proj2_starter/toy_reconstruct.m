function im_out = toy_reconstruct(toyim)
%toy_reconstruct(toyim)
%reconstrut image from gradient
[lh,lw] = size(toyim);

kPix = lh * lw;
kEdge = (lh-1) * (lw-1);
im2var = zeros(lh,lw);

im2var(1:kPix) = 1:kPix;
triplet = zeros(kEdge*4+1, 3);

b = zeros(2*kEdge+1,1);
%fill sparse matrix by triplet
e = 1;
t = 1;
for x=1:lw
   for y=1:lh
       if y < lh
           triplet(t,:) = [e,im2var(y,x),-1];
           triplet(t+1,:) = [e,im2var(y+1,x),1];
           b(e) = toyim(y+1,x) - toyim(y,x);
           t = t+2;
           e = e+1;
       end
       if x < lw
           triplet(t,:) = [e, im2var(y,x),-1];
           triplet(t+1,:) = [e, im2var(y,x+1), 1];
           b(e) = toyim(y,x+1) - toyim(y,x);
           t = t+2;
           e = e+1;
       end
   end
end

%border condition
triplet(t,:) = [e,1,1];
b(e) = toyim(1,1);

triplet2 = triplet(1:t, :);

%contruct sparse matrix and solve
A = sparse(triplet2(:,1), triplet2(:,2), triplet2(:,3), e, kPix);

im_out = zeros(lh, lw);
im_out(1:kPix) = A\b;

end