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
for x=1:lw-1
   for y=1:lh-1
       triplet(t,:) = [e,im2var(y,x),1];
       triplet(t+1,:) = [e,im2var(y+1,x),-1];
       triplet(t+2,:) = [e+kEdge, im2var(y,x),1];
       triplet(t+3,:) = [e+kEdge, im2var(y,x+1), -1];
       b(e) = toyim(y,x) - toyim(y+1,x);
       b(e+kEdge) = toyim(y,x) - toyim(y,x+1);
       t = t + 4;
       e = e + 1;
   end
end

%border condition
triplet(kEdge*4+1,:) = [kEdge+1,1,1];
b(kEdge+1) = toyim(1,1);

%contruct sparse matrix and solve
A = sparse(triplet(:,1), triplet(:,2), triplet(:,3), 2*kEdge+1, kPix);

im_out = zeros(lh, lw);
im_out(1:kPix) = lscov(A,b);

end