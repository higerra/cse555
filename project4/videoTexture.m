function videoTexture(filename)
%% videoTexture(filename)
vreader = VideoReader(filename);
w = vreader.Width;
h = vreader.Height;
fps = vreader.FrameRate;

maxKFrame = 1000;
kFrame = 0;
tempframes = cell(maxKFrame);
for i=1:maxKFrame
    if hasFrame(vreader)
        fprintf('Reading frame %d\n', i);
        tempframes{i} = im2double(readFrame(vreader));
        kFrame = kFrame+1;
    end
end
kFrame = kFrame - 1;
frames = cell(kFrame);
for i=1:kFrame
    frames{i} = tempframes{i+1};
end
fprintf('%d frames read\n', kFrame);

%compute initial D matrix
D = zeros(kFrame, kFrame);
for i=1:kFrame
    for j=1:kFrame
        I1 = reshape(frames{i}, 1, []);
        I2 = reshape(frames{j}, 1, []);
        D(i,j) = sqrt((I1-I2) * (I1-I2)');
    end
end

%diagnal filter
Df = zeros(size(D));
m = 2;
diaW = [1 2 2 1];
for i=m+1:kFrame-m+1
    for j=m+1:kFrame-m+1
      for k=-m:m-1
          Df(i,j) = Df(i,j) + diaW(k+m+1) * D(i+k,j+k);
      end
      Df(i,j) = Df(i,j) / sum(diaW);
    end
end
D = Df;

%Q-learning

%compute P matrix
aveD = mean(mean(D(m+1:kFrame-m,m+1:kFrame-m)));
sigma = 0.1 * aveD;
P = zeros(kFrame, kFrame);
for i=m+1:kFrame-m
    for j=m+1:kFrame-m
        P(i,j) = exp(-1*D(i+1,j)/sigma);
    end
   P(i,:) = P(i,:) / sum(P(i,:));
end

% Dv = D./max(max(D));
% figure(1),
% hold on;
% subplot(1,2,1);
% imshow(Dv);
% title('D matrix');
% subplot(1,2,2);
% Pv = P./max(max(P));
% imshow(Pv);
% title('P Matrix');
% hold off;

%cdf from pdf
cdf = cumsum(P,2);
source = m+1;
kOutput = 1000;

figure(2),
hold on;
x = rand(kOutput,1);
for i=1:kOutput
    %draw from distribution
    target = m+1;
    if x > cdf(source,1)
        for j=m+1:kFrame - m
            if x(i) > cdf(source,j-1) && x(i) <= cdf(source,j)
                target = j;
                break;
            end
        end
    end
    fprintf('Jump from frame %d to %d, p: %.3f\n', source, target, P(source,target));
    imshow(frames{target});
    source = target;
    pause(1.0/(fps+2));
end
hold off;
end
